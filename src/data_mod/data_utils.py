import os
from typing import Sequence, Dict, Union

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, default_collate
from torch.nn.utils.rnn import pad_sequence

import data_binding as data_binding

from .datasets import TYPE_TO_DATASET_CLASS, DatasetType
from .mmplanllm_dataset import MMPlanLLMDataset, MMPlanLLMModeSampler
from .multidataset import MultiDataset, MultiTaskSampler
from .multimode_dataset import MultiModeLazyDataset


def setup():
    # initialize the process group
    print("Initializing process group...")
    rank = int(os.environ['RANK'])
    if dist.is_initialized():
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
        dist.destroy_process_group()
    if rank == 0:
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")


def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    print("world size: ", world_size)
    print("local rank: ", local_rank)
    print("rank: ", rank)
    
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])


def get_dataset_type(dataset_type: Union[str, DatasetType]):
    return TYPE_TO_DATASET_CLASS[DatasetType(dataset_type).value]


def find_file_extension(file_path):
    valid_extensions = ['json', 'csv', 'tsv', 'txt']
    for ext in valid_extensions:
        if os.path.exists(f"{file_path}.{ext}"):
            return f"{file_path}.{ext}"

    if not os.path.exists(file_path):
        file_path = file_path.replace("_eval", "_val")
        for ext in valid_extensions:
            if os.path.exists(f"{file_path}.{ext}"):
                return f"{file_path}.{ext}"

    assert os.path.exists(file_path), file_path
    return file_path


def load_train_data(tokenizer: transformers.PreTrainedTokenizer, data_args: data_binding.DataArguments,
                    training_args: data_binding.TrainArgs, init_process_group=True, **kwargs):
    print(f"kwargs: {kwargs}")
    dataset_class = get_dataset_type(data_args.dataset_type)

    print("Loading train data...")
    train_data_path = data_args.data_path

    train_dataset = dataset_class(tokenizer=tokenizer, data_path=train_data_path, debug=training_args.debug, **kwargs)
    print("Train dataset size: ", len(train_dataset))

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print("world size: ", world_size, " rank: ", rank)

    if init_process_group:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=data_args.shuffle_data, drop_last=True)
    else:
        train_sampler = SequentialSampler(train_dataset)
    if isinstance(train_dataset, MMPlanLLMDataset):
        train_sampler = MMPlanLLMModeSampler(train_dataset, batch_size=training_args.per_device_train_batch_size)

    if training_args.evaluation_strategy != "no":
        print("Loading eval data...")
        eval_data_path = os.path.join(data_args.data_path, data_args.dataset_name,
                                      data_args.dataset_name + "_eval")
        eval_data_path = find_file_extension(eval_data_path)

        eval_dataset = dataset_class(tokenizer=tokenizer, data_path=eval_data_path, debug=training_args.debug,
                                     **kwargs)
        print("Eval dataset size: ", len(eval_dataset))

        if init_process_group:
            eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size, shuffle=data_args.shuffle_data, drop_last=True)
        else:
            eval_sampler = SequentialSampler(eval_dataset)
        if isinstance(train_dataset, MMPlanLLMDataset):
            eval_sampler = MMPlanLLMModeSampler(eval_dataset, batch_size=training_args.per_device_eval_batch_size)
    # if init_process_group:
    #     init_distributed()

    is_batch_sampler = isinstance(train_sampler, MultiTaskSampler)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1 if is_batch_sampler else training_args.per_device_train_batch_size,
        sampler=None if is_batch_sampler else train_sampler,
        batch_sampler=train_sampler if is_batch_sampler else None,
        num_workers=2 * torch.cuda.device_count(),
        pin_memory=True,
        collate_fn=lambda batch: padding_collate_fn(batch, padding_token=tokenizer.pad_token_id)
    )
    eval_dataloader = None
    if training_args.evaluation_strategy != "no":
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1 if is_batch_sampler else training_args.per_device_eval_batch_size,
            sampler=None if is_batch_sampler else eval_sampler,
            batch_sampler=eval_sampler if is_batch_sampler else None,
            num_workers=2 * torch.cuda.device_count(),
            pin_memory=True,
            collate_fn=lambda batch: padding_collate_fn(batch, padding_token=tokenizer.pad_token_id)
        )

    return train_dataloader, eval_dataloader


def load_test_dataset(tokenizer: transformers.PreTrainedTokenizer, infer_args: data_binding.InferenceArguments, data_args: data_binding.DataArguments,
                      model_args: data_binding.ModelArguments, **kwargs):

    dataset_class = get_dataset_type(data_args.dataset_type)

    print("Loading test data...")
    test_dataset = dataset_class(tokenizer=tokenizer, data_path=infer_args.test_file, debug=False,
                                  model_name=model_args.ckpt_path, **kwargs)
    return test_dataset


def padding_collate_fn(batch, padding_token=0):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    position_ids = [item['position_ids'] for item in batch] if 'position_ids' in batch[0] else None
    pixel_values = [item['pixel_values'] for item in batch] if 'pixel_values' in batch[0] else None

    # Pad sequences
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_token)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    if position_ids is not None:
        padded_position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
    if pixel_values is not None:
        # pixel values has shape (batch_size, num_images, num_channels, height, width)
        # we pad the number of images to the maximum number of images in the batch
        # print("pixel_values.shape: ", pixel_values[0].shape)
        max_num_images = max([item.shape[0] for item in pixel_values])
        padded_pixel_values = torch.zeros((len(batch), max_num_images, pixel_values[0].shape[1], pixel_values[0].shape[2], pixel_values[0].shape[3]), dtype=pixel_values[0].dtype)
        for i, item in enumerate(pixel_values):
            padded_pixel_values[i, :item.shape[0], :, :, :] = item

    # Replace the original sequences with the padded ones
    for i, item in enumerate(batch):
        item['input_ids'] = padded_input_ids[i]
        item['labels'] = padded_labels[i]
        item['attention_mask'] = padded_attention_masks[i]
        if position_ids is not None:
            item['position_ids'] = padded_position_ids[i]
        if pixel_values is not None:
            item['pixel_values'] = padded_pixel_values[i]

    return default_collate(batch)
