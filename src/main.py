import ast
import json
import os.path
import shutil
from typing import Dict

import torch
import torch.distributed as dist
import transformers
import torch.backends.cudnn as cudnn
from torch import nn
from transformers import AutoTokenizer

from constants import *
from data_mod import data_utils
from data_binding import ModelArguments, DataArguments, OptimizerArguments, TrainArgs, PrecisionType, DPOArguments, \
    LoRaArguments

from src.models.model import FrozenArgs, VIGIA, VIGIAConfig

from trainers.vigia_trainer import VigiaTrainer
from trainers.dual_dataset_trainer import DualDatasetTrainer
from trainers import trainer_utils



def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def recursive_conversion(module: nn.Module, dtype: torch.dtype):
    """Recursively converts a module and its submodules to the specified dtype."""
    def convert_module(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            if m.weight.dtype != dtype: 
                m.weight.data = m.weight.data.to(dtype)
            if hasattr(m, 'bias') and m.bias is not None and m.bias.dtype != dtype:
                m.bias.data = m.bias.data.to(dtype)

    module.apply(convert_module)
    return module

def zero_out_embedding_gradients_except_tokens(model, token_ids):
    """
    Zero out the gradients of the embedding layer except for the specified tokens.
    
    Args:
        model: The language model
        token_ids: List of token IDs to preserve gradients for
    """
    # Convert to list if a single integer is passed
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    
    # Ensure token_ids is a list
    token_ids = list(token_ids)
    
    def hook_fn(grad):
        # Create a mask of zeros with the same shape as the gradient
        mask = torch.zeros_like(grad)
        
        # Set the positions corresponding to the specified tokens to 1
        for token_id in token_ids:
            mask[token_id] = 1.0
        
        # Apply the mask to only keep gradients for the specified tokens
        return grad * mask
    
    # Find the embedding layer - this may need adjustment based on your model architecture
    if hasattr(model, 'model') and hasattr(model.model, 'lm'):
        # For your VIGIA model structure
        if hasattr(model.model.lm, 'get_input_embeddings'):
            embedding_layer = model.model.lm.get_input_embeddings()
        else:
            # Fallback for models where the embedding might be directly accessible
            embedding_layer = model.model.lm.transformer.wte
    elif hasattr(model, 'get_input_embeddings'):
        # For HuggingFace models
        embedding_layer = model.get_input_embeddings()
    else:
        raise ValueError("Could not locate embedding layer. Please adjust the code for your model architecture.")
    
    # Register the gradient hook
    if embedding_layer.weight.requires_grad:
        embedding_layer.weight.register_hook(hook_fn)
        print(f"Registered gradient hook on embedding layer. Only gradients for token IDs {token_ids} will be updated.")
    else:
        print("Warning: Embedding layer does not require gradients. No hook registered.")
    
    return model


def load_model(model_args: FrozenArgs, ckpt_path=None, seq_max_length=DEFAULT_MAX_LEN, parallel_type: str = "NO", dtype=torch.float32, lora_args: LoRaArguments = None):

    tokenizer = AutoTokenizer.from_pretrained(model_args.text_decoder, use_fast=True, trust_remote_code=True)
    # Add an image token for loss masking (and visualization) purposes.
    tokenizer.add_special_tokens({"cls_token": IMG_TOKEN})  # add special image token to tokenizer
    tokenizer.add_tokens([RET_TOKEN, RET_TOKEN_2])  # add special retrieval tokens to tokenizer
    start_ret_token_idx = tokenizer.get_vocab().get(RET_TOKEN, None)
    end_ret_token_idx = tokenizer.get_vocab().get(RET_TOKEN_2, None)
    assert start_ret_token_idx is not None and isinstance(start_ret_token_idx, int), 'Retrieval token not found in tokenizer vocab.'
    assert end_ret_token_idx is not None and isinstance(end_ret_token_idx, int), 'Retrieval token not found in tokenizer vocab.'
    
    model_args.start_ret_token_id = start_ret_token_idx
    model_args.end_ret_token_id = end_ret_token_idx
    model_args.img_token_id = tokenizer.cls_token_id

    # Check for padding token and handle if missing
    if tokenizer.pad_token_id is None:
        print("Padding token is missing. Setting pad token to eos token as default value.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print(f"Using existing padding token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    # Check if eos token exists
    if tokenizer.eos_token_id is None:
        raise ValueError("Model has no end-of-sequence token defined. This is required for proper functioning.")
    else:
        print(f"Using end-of-sequence token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    model_args.pad_token_id = tokenizer.pad_token_id
    model_args.eos_token_id = tokenizer.eos_token_id

    # print the id of the special tokens we added
    print(f"IMG token: {IMG_TOKEN} - {tokenizer.cls_token_id}")
    print(f"start RET token: {RET_TOKEN} - {start_ret_token_idx}")
    print(f"end RET token: {RET_TOKEN_2} - {end_ret_token_idx}")
    print(f"PAD token: {tokenizer.pad_token} - {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token} - {tokenizer.eos_token_id}")

    # TODO: review truncation side. On LazyOldDataset, we truncate from the right side to keep the caption making sense, and add the RET token to the left side if it's not there
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left' # truncate from left side because otherwise the retrieval token will be removed

    tokenizer.model_max_length = seq_max_length

    # TODO: if loading from checkpoint, this needs to come first as the text_decoder might be different
    if ckpt_path is None:
        model = VIGIA(config=VIGIAConfig(**model_args.to_dict(), vocab_size=len(tokenizer)))
    else:
        model = VIGIA.from_pretrained(ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True, trust_remote_code=True)
        # updated the pad and eos tokens
        model.model.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.eos_token_id = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.model.start_ret_token_id = start_ret_token_idx
        model.model.end_ret_token_id = end_ret_token_idx
        # update the tokenizer
        tokenizer.pad_token_id = model.model.pad_token_id
        tokenizer.eos_token_id = model.model.eos_token_id
        tokenizer.model_max_length = seq_max_length

        # set require grad properly based on model_args
        model.update_freezes(
            freeze_lm=model_args.freeze_lm,
            freeze_vm=model_args.freeze_vm,
            freeze_cap=model_args.freeze_cap,
            freeze_ret=model_args.freeze_ret,
        )

        assert model.config.vocab_size == len(tokenizer), f"Vocab size mismatch between model and tokenizer ({model.config.vocab_size} vs {len(tokenizer)})"
        assert tokenizer.get_vocab().get(RET_TOKEN, None) == model.model.start_ret_token_id, f"Retrieval token ID mismatch between model and tokenizer ({model.model.start_ret_token_id} vs {tokenizer.get_vocab().get(RET_TOKEN, None)})"
        assert tokenizer.get_vocab().get(RET_TOKEN_2, None) == model.model.end_ret_token_id, f"Retrieval token ID mismatch between model and tokenizer ({model.model.end_ret_token_id} vs {tokenizer.get_vocab().get(RET_TOKEN_2, None)})"
        assert tokenizer.cls_token_id == model.model.img_token_id, f"Image token ID mismatch between model and tokenizer ({model.model.img_token_id} vs {tokenizer.cls_token_id})"

    # update the input_embeddings with the new vocab size
    # model.model.lm.resize_token_embeddings(len(tokenizer))
    tokenizer.chat_template = CHAT_TEMPLATE

    if lora_args:
        if lora_args.lora_merge_adapter:
            print("Merging adapter into model")
            model.model.merge_lm_lora()
            model.config.use_lora_on_lm = False
        if lora_args.lora:
            success = model.model.make_lm_lora(lora_rank=lora_args.lora_rank, lora_alpha=lora_args.lora_alpha, lora_dropout=lora_args.lora_dropout)
            model.config.use_lora_on_lm = True
            model.config.lm_lora_rank = lora_args.lora_rank
            model.config.lm_lora_alpha = lora_args.lora_alpha
            model.config.lm_lora_dropout = lora_args.lora_dropout
            if not success:
                print("Failed to make LM LoRA, continuing...")

    model = model.to(dtype)
    model = recursive_conversion(model, dtype)


    if parallel_type == "DP" and torch.cuda.device_count() > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        # get model on local rank device
        model = model.to(f"cuda:{local_rank}")

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = model.cuda()
        model.model.lm = model.model.lm.cuda()

    model = model.to(dtype)

    # Enable cuDNN auto-tuner to find the best algorithm for the hardware
    cudnn.benchmark = True

    return model, tokenizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    # parse arguments
    parser = transformers.HfArgumentParser(( FrozenArgs, OptimizerArguments, ModelArguments, DataArguments, TrainArgs, DPOArguments, LoRaArguments))
    vigia_args, optim_args, model_args, data_args, training_args, dpo_args, lora_args = parser.parse_args_into_dataclasses()  # type: (FrozenArgs, OptimizerArguments, ModelArguments, DataArguments, TrainArgs, DPOArguments, LoRaArguments)

    training_args.to_dict()
    optim_args.to_dict()
    model_args.to_dict()
    data_args.to_dict()
    dpo_args.to_dict()
    lora_args.to_dict()

    config = training_args.to_dict() | optim_args.to_dict() | model_args.to_dict() | data_args.to_dict() | dpo_args.to_dict() | lora_args.to_dict()

    print(training_args)
    data_args.dataset_kwargs = ast.literal_eval(data_args.dataset_kwargs)

    # Build output_dir if not provided
    if model_args.ckpt_path and training_args.resume_from_checkpoint:
        training_args.output_dir = os.path.dirname(model_args.ckpt_path)
        print("Resuming from checkpoint so setting output_dir to ", training_args.output_dir)
        if training_args.resume_from_step == 0:
            training_args.resume_from_step = int(os.path.basename(model_args.ckpt_path).replace("checkpoint_", ""))
    elif training_args.output_dir is not None and training_args.output_dir != "":
        training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name).__str__()
    else:
        training_args.output_dir = os.path.join("/experiments", training_args.project_name,
                                                training_args.run_name).__str__()

    if training_args.resume_from_checkpoint:
        print(f"Resuming from step {training_args.resume_from_step}")

    # Get current GPU and GPU_COUNT
    local_rank = int(os.environ['LOCAL_RANK'] if 'LOCAL_RANK' in os.environ else 0)
    rank = int(os.environ['RANK'] if 'RANK' in os.environ else 0)
    world_size = int(os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ else torch.cuda.device_count())

    torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))

    # Check if output_dir exists, if so delete or throw warning.
    if rank == 0:
        if not training_args.resume_from_checkpoint:
            if os.path.exists(training_args.output_dir):
                if training_args.overwrite_output_dir:
                    print(f"WARNING: Directory {training_args.output_dir} already exists. OVERWRITING")
                    shutil.rmtree(training_args.output_dir)
                else:
                    print(f"ERROR: Directory {training_args.output_dir} already exists.")
                    return

            # Create output dir
            os.makedirs(training_args.output_dir, exist_ok=True)

            # save training arguments
            json.dump(config, open(f"{training_args.output_dir}/run_arguments.json", "w"), indent=4)

            # Save model args to disk.
            with open(os.path.join(training_args.output_dir, 'model_args.json'), 'w') as f:
                json.dump(vars(vigia_args), f, indent=4)

        else:
            if not os.path.exists(training_args.output_dir):
                print(f"ERROR: Directory {training_args.output_dir} does not exist.")
                return

    # set manual seed
    torch.manual_seed(training_args.seed)
    dtype = torch.float16 if training_args.load_dtype == PrecisionType.FP16 \
        else torch.bfloat16 if training_args.load_dtype == PrecisionType.BF16 \
        else torch.float32

    print(f"Using {dtype} precision")

    if training_args.parallel_type == "DP" and world_size > 1:
        print("###LOG: Using Distributed Data Parallel, initializing process group")
        data_utils.init_distributed()

    # Loading model and tokenizer
    model, tokenizer = load_model(vigia_args, ckpt_path=model_args.ckpt_path, parallel_type=training_args.parallel_type,
                                      seq_max_length=training_args.seq_max_length, dtype=dtype, lora_args=lora_args)

    if local_rank == 0:
        for name, param in model.named_parameters():
            if param.dtype != dtype:
                print(f"Layer {name} is in {param.dtype}")

        for name, buffer in model.named_buffers():
            if buffer.dtype != dtype:
                print(f"Buffer {name} is in {buffer.dtype}")

    original_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print model params and if they require grad
    if rank == 0:
        print()
        print(trainer_utils.get_params_count_str(model, trainable_only=True))
        print()
    # Load dataset
    data_args.dataset_kwargs['feature_extractor_model'] = vigia_args.visual_encoder  # add vision encoder to dataset kwargs
    data_args.dataset_kwargs['batch_size'] = training_args.per_device_train_batch_size * world_size
    train_dataloader, eval_dataloader = data_utils.load_train_data(
        tokenizer, data_args, training_args,
        init_process_group=world_size > 1,
        model_name=vigia_args.text_decoder if model_args.ckpt_path is None else model_args.ckpt_path,
        base_model=vigia_args.text_decoder,
        **data_args.dataset_kwargs,
        )

    print("Train dataloader size: ", len(train_dataloader))
    if training_args.dual_dataset:
        print(f"Loading second dataset")
        if data_args.second_dataset_name is None:
            print("Warning: Second dataset name is not provided. Using the primary dataset name for the second dataset.")
            data_args.second_dataset_name = data_args.dataset_name
        if data_args.second_dataset_type is None:
            print("Warning: Second dataset type is not provided. Using the primary dataset type for the second dataset.")
            data_args.second_dataset_type = data_args.dataset_type
        # load_train_data uses the first dataset values by default, so we override them here
        data_args.dataset_name = data_args.second_dataset_name
        data_args.dataset_type = data_args.second_dataset_type
        data_args.data_path = data_args.second_data_path
        training_args.per_device_train_batch_size = training_args.per_device_second_train_batch_size
        data_args.second_dataset_kwargs = ast.literal_eval(data_args.second_dataset_kwargs)
        data_args.second_dataset_kwargs['feature_extractor_model'] = vigia_args.visual_encoder
        data_args.second_dataset_kwargs['batch_size'] = training_args.per_device_second_train_batch_size * world_size
        second_train_dataloader, second_eval_dataloader = data_utils.load_train_data(
                tokenizer, data_args, training_args,
                init_process_group=world_size > 1,
                model_name=vigia_args.text_decoder if model_args.ckpt_path is None else model_args.ckpt_path,
                base_model=vigia_args.text_decoder,
                **data_args.second_dataset_kwargs,
                )
        train_dataloader = [train_dataloader, second_train_dataloader]
        eval_dataloader = [eval_dataloader, second_eval_dataloader]
        print("Second train dataloader size: ", len(second_train_dataloader))
    
    trainer_class = DualDatasetTrainer if training_args.dual_dataset else VigiaTrainer

    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=None,
        args=training_args,
        data_args=data_args,
        config=config,
    )

    trainer.train()
    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        cleanup()
        raise e
    
