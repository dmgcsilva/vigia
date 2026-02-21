from dataclasses import asdict, dataclass, field, fields
from io import BytesIO
from typing import Callable, List, Optional, Tuple, Union, Dict
from collections import namedtuple, OrderedDict
import json
import numpy as np
import os

import requests
import torch
from info_nce import info_nce
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoFeatureExtractor, AutoTokenizer
from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import CLIPVisionModel, CLIPVisionConfig, SiglipVisionModel
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from .projectors import CONNECTOR_REGISTRY, RETRIEVAL_PROJECTOR_REGISTRY

import torch

SEP_TOKEN = "\n"
IMG_TOKEN = "[IMG]"
IMG_START_TOKEN = "<im_start>"
IMG_END_TOKEN = "<im_end>"
RET_TOKEN = "[RET]"
RET_TOKEN_2 = "[RET2]"

IGNORE_INDEX = -100

# source: https://blog.eleuther.ai/rotary-embeddings/
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        # Don't store these as module attributes to avoid device sync issues
        self.cos_sin_cache = {}

    def _get_cos_sin_cache(self, device):
        """Get device-specific cos/sin cache, compute if needed"""
        if device not in self.cos_sin_cache:
            t = torch.arange(self.max_seq_len, device=device).type_as(self.inv_freq.to(device))
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_sin_cache[device] = (
                emb.cos()[None, :, None, :],  # (1, max_seq_len, 1, dim)
                emb.sin()[None, :, None, :]   # (1, max_seq_len, 1, dim)
            )
        return self.cos_sin_cache[device]

    def forward(self, x, offset=None):
        # x.shape: (bs, seq_len, hidden_dim, dim)
        # offset.shape: (bs, 1) or None
        batch_size, seq_len, hidden_dim, dim = x.shape

        cached_cos, cached_sin = self._get_cos_sin_cache(x.device)

        if offset is None:
            offset = torch.zeros((batch_size, 1), device=x.device, dtype=torch.long)
        else:
            offset = offset.unsqueeze(1).to(torch.long)  # Ensure correct type

        # Generate position indices for each element in the sequence
        seq_positions = torch.arange(seq_len, device=x.device).view(1, seq_len)  # [1, seq_len]
        seq_positions = seq_positions.expand(batch_size, -1)  # [batch_size, seq_len]
        # print("seq_positions.shape", seq_positions.shape)
        # Add offsets to each position
        positions = seq_positions + offset  # [batch_size, seq_len]
        # print("positions 1", positions.shape)
        # Clamp positions to max_seq_len - 1
        positions = torch.clamp(positions, max=self.max_seq_len - 1)
        
        positions_flat = positions.view(-1)  # Flatten to [batch_size * seq_len]
        # print("positions 2", positions.shape)
        # Select appropriate rows from cached cos/sin
        cos_selected = torch.index_select(cached_cos[0], 0, positions_flat)  # [batch_size*seq_len, 1, dim]
        sin_selected = torch.index_select(cached_sin[0], 0, positions_flat)  # [batch_size*seq_len, 1, dim]
        # print("cos_selected.shape", cos_selected.shape, "sin_selected.shape", sin_selected.shape)
        # print("x.shape", x.shape, 'offset.shape', offset.shape)
        # Reshape to match input dimensions
        cos_values = cos_selected.view(batch_size, seq_len, 1, dim).expand(-1, -1, hidden_dim, -1)
        sin_values = sin_selected.view(batch_size, seq_len, 1, dim).expand(-1, -1, hidden_dim, -1)
        
        # Apply rotary transformation
        x_rotated = (x * cos_values) + (rotate_half(x) * sin_values)
        return x_rotated


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


@dataclass
class FrozenArgs:
    freeze_lm: bool = field(default=True, metadata={"help": "Whether to freeze the language model."})
    freeze_emb: bool = field(default=False, metadata={"help": "Whether to freeze the embeddings, regardless of the freeze_lm flag."})
    freeze_vm: bool = field(default=True, metadata={"help": "Whether to freeze the vision model."})
    freeze_cap: bool = field(default=False, metadata={"help": "Whether to freeze the captioning layers."})
    freeze_ret: bool = field(default=False, metadata={"help": "Whether to freeze the retrieval layers, also keeps the llm embedding trainable."})

    text_decoder: str = field(default='facebook/opt-6.7b', metadata={"help": "The name of the text encoder model."})
    visual_encoder: str = field(default='openai/clip-vit-large-patch14', metadata={"help": "The name of the visual encoder model."})
    n_visual_tokens: int = field(default=1, metadata={"help": "The number of visual tokens."})
    image_embed_dropout_prob: float = field(default=0.0, metadata={"help": "The dropout probability for the image embedding."})
    task: str = field(default='captioning', metadata={"help": "The task to perform."})
    shared_emb_dim: int = field(default=256, metadata={"help": "The dimension of the shared embedding."})
    text_embed_dropout_prob: float = field(default=0.0, metadata={"help": "The dropout probability for the text embedding."})
    start_ret_token_id: int = field(default=0, metadata={"help": "The index of the retrieval token."})
    end_ret_token_id: int = field(default=0, metadata={"help": "The index of the retrieval token."})
    img_token_id: int = field(default=0, metadata={"help": "The index of the image token."})
    pad_token_id: int = field(default=-1, metadata={"help": "The index of the padding token."})
    eos_token_id: int = field(default=-1, metadata={"help": "The index of the end-of-sequence token."})

    projector_type: str = field(default='linear', metadata={"help": "The type of projector to use."})
    connector_type: str = field(default='linear', metadata={"help": "The type of connector to use."})
    use_cls_token: bool = field(default=True, metadata={"help": "Whether to use the CLS token."})

    cap_loss_scale: float = field(default=1.0, metadata={"help": "The scaling factor for the captioning loss."})
    ret_loss_scale: float = field(default=1.0, metadata={"help": "The scaling factor for the retrieval loss."})

    use_negatives: bool = field(default=False, metadata={"help": "Whether to use negative sampling."})
    negative_count: int = field(default=512, metadata={"help": "The number of negative samples to use."})

    use_pos_emb: bool = field(default=False, metadata={"help": "Whether to use positional embeddings."})

    @staticmethod
    def from_dict(d: Dict):
        return FrozenArgs(**d)

    def to_dict(self):
        return asdict(self)
    

class VIGIAConfig(PretrainedConfig):
    model_type = "revistamodel"

    def __init__(self,
                freeze_lm: bool = True,
                freeze_emb: bool = False,
                freeze_vm: bool = True,
                freeze_cap: bool = False,
                freeze_ret: bool = False,
                text_decoder: str = 'facebook/opt-6.7b',
                visual_encoder: str = 'openai/clip-vit-large-patch14',
                n_visual_tokens: int = 1,
                image_embed_dropout_prob: float = 0.0,
                task: str = 'captioning',
                shared_emb_dim: int = 256,
                text_embed_dropout_prob: float = 0.0,
                start_ret_token_id: int = 0,
                end_ret_token_id: int = 0,
                img_token_id: int = 0,
                pad_token_id: int = -1,
                eos_token_id: int = -1,
                cap_loss_scale: float = 1.0,
                ret_loss_scale: float = 1.0,
                use_negatives: bool = False,
                negative_count: int = 512,
                use_pos_emb: bool = False,
                vocab_size: int = -1,
                projector_type: str = 'linear',
                connector_type: str = 'linear',
                use_cls_token: bool = True,
                **kwargs):
        self.freeze_lm = freeze_lm
        self.freeze_emb = freeze_emb
        self.freeze_vm = freeze_vm
        self.freeze_cap = freeze_cap
        self.freeze_ret = freeze_ret
        self.text_decoder = text_decoder
        self.visual_encoder = visual_encoder
        self.n_visual_tokens = n_visual_tokens
        self.image_embed_dropout_prob = image_embed_dropout_prob
        self.task = task
        self.shared_emb_dim = shared_emb_dim
        self.text_embed_dropout_prob = text_embed_dropout_prob
        self.start_ret_token_id = start_ret_token_id
        self.end_ret_token_id = end_ret_token_id
        self.img_token_id = img_token_id
        self.cap_loss_scale = cap_loss_scale
        self.ret_loss_scale = ret_loss_scale
        self.use_negatives = use_negatives
        self.negative_count = negative_count
        self.use_pos_emb = use_pos_emb
        self.vocab_size = vocab_size
        self.projector_type = projector_type
        self.connector_type = connector_type
        self.use_cls_token = use_cls_token
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class VIGIAModel(nn.Module):
    def __init__(self, config: VIGIAConfig): # Removed args
        super().__init__()
        self.config = config # Now uses config

        text_decoder = config.text_decoder
        visual_encoder = config.visual_encoder
        n_visual_tokens = config.n_visual_tokens
        print(f"Using {text_decoder} for the language model.")
        print(f"Using {visual_encoder} for the visual model with {n_visual_tokens} visual tokens.")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if 'facebook/opt' in text_decoder:
            self.lm = OPTForCausalLM.from_pretrained(text_decoder)
        else:
            try:
                self.lm = AutoModelForCausalLM.from_pretrained(text_decoder)
            except  Exception as e:
                print(f'Error loading model {text_decoder}: {e}')
                raise e

        self.text_decoder = text_decoder

        # NOTE: Resizing sets all token embeddings and all lm_head weights (since they are tied in OPT)
        # to be trainable (param.requires_grad = True).
        self.start_ret_token_id = config.start_ret_token_id
        self.end_ret_token_id = config.end_ret_token_id
        print(f'Initializing embedding for the retrieval token [RET] (start id = {self.start_ret_token_id}, end id = {self.end_ret_token_id}).')
        if self.config.vocab_size == -1:
            config.vocab_size = self.lm.config.vocab_size
        self.lm.resize_token_embeddings(self.config.vocab_size)
        if 'qwen' in text_decoder.lower() and '7b' in text_decoder.lower():
            print("Warning: Using Qwen 7B model, manually setting the embedding for the [RET] and [IMG] tokens, to the average of the other tokens.")
            mean_emb = self.lm.get_input_embeddings().weight.mean(dim=0)
            self.lm.get_input_embeddings().weight[self.start_ret_token_id] = mean_emb
            self.lm.get_input_embeddings().weight[self.end_ret_token_id] = mean_emb
            self.lm.get_input_embeddings().weight[config.img_token_id] = mean_emb

        print("Restoring pretrained weights for the visual model.")
        if 'clip' in visual_encoder:
            self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder)
        elif "siglip" in visual_encoder:
            self.visual_model = SiglipVisionModel.from_pretrained(visual_encoder)
        else:
            self.visual_model = AutoModel.from_pretrained(visual_encoder)

        if 'clip' in visual_encoder or "siglip" in visual_encoder:
            hidden_size = self.visual_model.config.hidden_size
        else:
            raise NotImplementedError

        self.visual_model_name = visual_encoder

        embedding_dim = self.input_embeddings.embedding_dim * self.config.n_visual_tokens
        out_dim = self.config.shared_emb_dim

        if "opt" in text_decoder:
            in_dim = self.lm.config.word_embed_proj_dim
        else:
            in_dim = self.lm.config.hidden_size

        self.ret_text_to_img = RETRIEVAL_PROJECTOR_REGISTRY[self.config.projector_type](in_dim=in_dim, out_dim=out_dim)

        self.visual_embeddings = CONNECTOR_REGISTRY[self.config.connector_type](in_dim=hidden_size, out_dim=embedding_dim)
        self.visual_fc = nn.Linear(hidden_size, out_dim)

        self.image_dropout = nn.Dropout(self.config.image_embed_dropout_prob)

        self.negative_queue_text = None
        self.negative_queue_image = None
        self.max_queue_size = self.config.negative_count

        self.img_token_id = config.img_token_id
        self.pad_token_id = config.pad_token_id if config.pad_token_id != -1 and config.pad_token_id is not None else self.lm.config.pad_token_id
        self.eos_token_id = config.eos_token_id if config.eos_token_id != -1 and config.eos_token_id is not None  else self.lm.config.eos_token_id

        if self.config.use_pos_emb:
            self.pos_embs = Rotary(out_dim)

        # freeze model params according to the config
        if self.config.freeze_lm:
            self.lm.eval()
            print("Freezing the LM.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

        if self.config.freeze_vm:
            print("Freezing the VM.")
            self.visual_model.eval()
            for param in self.visual_model.parameters():
                param.requires_grad = False
        else:
            self.visual_model.train()


        if self.config.freeze_cap:
            print("Freezing the captioning layers.")
            self.visual_embeddings.eval()
            for param in self.visual_embeddings.parameters():
                param.requires_grad = False
        else:
            self.visual_embeddings.train()

        if self.config.freeze_ret:
            print("Freezing the retrieval layers.")
            self.ret_text_to_img.eval()
            for param in self.ret_text_to_img.parameters():
                param.requires_grad = False 
            self.visual_fc.eval()
            for param in self.visual_fc.parameters():
                param.requires_grad = False
            self.logit_scale.requires_grad_(False)
        else:
            self.ret_text_to_img.train()
            self.visual_fc.train()
            self.logit_scale.requires_grad_(True)
            self.input_embeddings.requires_grad_(True)
            if not self.lm.config.tie_word_embeddings:
                print("Word embeddings are not tied to the output embeddings, setting output embeddings to be trainable.")
                self.output_embeddings.weight.requires_grad_(True)

    @property
    def input_embeddings(self):
        return self.lm.get_input_embeddings()
    
    @property
    def output_embeddings(self):
        return self.lm.get_output_embeddings()

    def make_lm_lora(self, lora_rank: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.03):

        if isinstance(self.lm, LoraModel) or isinstance(self.lm, PeftModel):
            print("Model already has a LoraModel or PeftModel, skipping.")
            return False

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj'],
        )
        self.lm = get_peft_model(self.lm, peft_config)

        # update the input embeddings
        # self.input_embeddings = self.lm.get_input_embeddings()
        self.input_embeddings.requires_grad_(True if not self.config.freeze_lm or not self.config.freeze_ret else False)
        if not self.lm.config.tie_word_embeddings:
            print("Word embeddings are not tied to the output embeddings, setting output embeddings to be trainable.")
            # self.output_embeddings = self.lm.get_output_embeddings()
            self.output_embeddings.weight.requires_grad_(True if not self.config.freeze_lm or not self.config.freeze_ret else False)

        print("---> Converted LM to LoRA")
        return True

    def merge_lm_lora(self):
        if isinstance(self.lm, LoraModel) or isinstance(self.lm, PeftModel):
            print("Merging adapter")
            self.lm = self.lm.merge_and_unload()
            if not self.config.freeze_lm:
                # set all parameters trainable
                for param in self.lm.parameters():
                    param.requires_grad = True

            # self.input_embeddings = self.lm.get_input_embeddings()
            self.input_embeddings.requires_grad_(True)
            if not self.lm.config.tie_word_embeddings:
                print("Word embeddings are not tied to the output embeddings, setting output embeddings to be trainable.")
                # self.output_embeddings = self.lm.get_output_embeddings()
                self.output_embeddings.weight.requires_grad_(True)

            print("---> Merged LM and LoRA")
        else:
            print(f"WARNING: Can't merge, model is not a LoraModel or PeftModel, but a {type(self.lm)}")

    def get_visual_embs(self, pixel_values: torch.FloatTensor, mode: str = 'captioning', position_offset: Optional[torch.LongTensor] = None):
        if mode not in ['captioning', 'retrieval']:
            raise ValueError(f'mode should be one of ["caption", "retrieval"], got {mode} instead.')

        # Extract visual embeddings from the vision encoder.
        if 'clip' in self.visual_model_name or "siglip" in self.visual_model_name:
            outputs = self.visual_model(pixel_values)
            if mode == "retrieval" or self.config.use_cls_token:
                encoder_outputs = outputs.pooler_output
            else: # get the last hidden state
                try:
                    encoder_outputs = outputs.hidden_states[-1]
                except:
                    encoder_outputs = outputs.last_hidden_state
        else:
            raise NotImplementedError

        # Use the correct fc based on function argument.
        if mode == 'captioning':
            visual_embs = self.visual_embeddings(encoder_outputs)  # (2, D * n_visual_tokens)
            if self.config.use_cls_token:
                visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.config.n_visual_tokens, -1))
        elif mode == 'retrieval':
            visual_embs = self.visual_fc(encoder_outputs)  # (2, D * n_visual_tokens)
            if position_offset is None:
                position_offset = torch.zeros((visual_embs.shape[0], 1), device=visual_embs.device, dtype=torch.long)
            if self.config.use_pos_emb:  #TODO review this
                visual_embs = self.pos_embs(visual_embs.unsqueeze(1).unsqueeze(1), offset=position_offset)
                visual_embs = visual_embs.squeeze(1).squeeze(1)
            visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
        else:
            raise NotImplementedError

        visual_embs = self.image_dropout(visual_embs)
        return visual_embs

    def remove_img_tokens(self, labels: torch.LongTensor, embs: torch.FloatTensor = None, last_embedding_idx: torch.LongTensor = None):

        was_unbatched = False
        if len(labels.shape) == 1:
            # unbatched case
            labels = labels.unsqueeze(0)
            was_unbatched = True

        new_labels = []
        if embs is not None:
            new_embs = []
        if last_embedding_idx is not None:
            new_last_embedding_idx = []

        for i in range(labels.shape[0]):
            # get the index of all the image tokens in the labels
            label = labels[i]
            img_idxs = (label == self.img_token_id).nonzero(as_tuple=True)

            if embs is not None:
                emb = embs[i]
            if last_embedding_idx is not None:
                last_idx = last_embedding_idx[i]
            # Check if the tensor is empty
            if len(img_idxs) > 0 and img_idxs[0].shape[0] > 0:
                for j in range(len(img_idxs)):
                    # remove the image tokens from the labels
                    label = torch.cat([label[:img_idxs[j]], label[img_idxs[j] + 1:]])
                    if embs is not None:
                        emb = torch.cat([emb[:img_idxs[j]], emb[img_idxs[j] + 1:]])
                    if last_embedding_idx is not None:
                        last_idx -= 1

            new_labels.append(label)
            if embs is not None:
                new_embs.append(emb)
            if last_embedding_idx is not None:
                new_last_embedding_idx.append(last_idx)

        new_labels = torch.stack(new_labels, dim=0)
        new_embs = torch.stack(new_embs, dim=0) if embs is not None else None
        new_last_embedding_idx = torch.stack(new_last_embedding_idx, dim=0) if last_embedding_idx is not None else None

        if was_unbatched:
            new_labels = new_labels.squeeze(0)

        return new_labels, new_embs, new_last_embedding_idx

    def train(self, mode=True):
        super(VIGIAModel, self).train(mode=mode)
        # Overwrite train() to ensure Frozen models remain frozen.
        if self.config.freeze_lm:
            self.lm.eval()
            if not self.config.freeze_ret:
                self.input_embeddings.requires_grad_(True)
                if not self.lm.config.tie_word_embeddings:
                    self.output_embeddings.weight.requires_grad_(True)
        if self.config.freeze_vm:
            self.visual_model.eval()
        if self.config.freeze_cap:
            self.visual_embeddings.eval()
        if self.config.freeze_ret:
            self.ret_text_to_img.eval()
            self.visual_fc.eval()
            self.logit_scale.requires_grad_(False)

    def _forward_captioning(
            self,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            input_embs: torch.FloatTensor,
            visual_embs: torch.FloatTensor,
            concat_captions: bool = False,
    ):

        batch_size, sequence_length, dimension = input_embs.shape
        _, frames_count, visual_sequence_length, _ = visual_embs.shape
        
        new_embs_list = []
        full_labels = []
        for i in range(batch_size):
            sample_input_ids = input_ids[i]
            sample_input_embs = input_embs[i]
            sample_labels = labels[i]
            sample_visual_embs = visual_embs[i]

            special_token_positions = (sample_input_ids == self.img_token_id).nonzero(as_tuple=True)[0]
            new_sample_embs = []
            new_sample_labels = []
            last_pos = 0

            frame_idx = 0

            for pos in special_token_positions:
                new_sample_embs.append(sample_input_embs[last_pos:pos])
                new_sample_labels.append(sample_labels[last_pos:pos])
                if frame_idx < frames_count:
                    new_sample_embs.append(sample_visual_embs[frame_idx])
                    new_sample_labels.append(torch.full((visual_sequence_length,), -100, dtype=torch.long, device=sample_labels.device))
                else:
                    raise ValueError(f"Frame index {frame_idx} is out of bounds for the visual embeddings with shape {visual_embs.shape}. Found {self.img_token_id} at postions {special_token_positions}.")

                last_pos = pos + 1
                frame_idx += 1

            new_sample_embs.append(sample_input_embs[last_pos:])
            new_sample_labels.append(sample_labels[last_pos:])

            # Concatenate all parts for this sample.
            new_sample_embs = torch.cat(new_sample_embs, dim=0)
            new_sample_labels = torch.cat(new_sample_labels, dim=0)
            new_embs_list.append(new_sample_embs)
            full_labels.append(new_sample_labels)

        # Pad the sequences to the maximum length. This is copout solution, but it works. Ideally I would take the established route of adding placeholders to the input_ids and labels through the tokenizer/processor.
        max_len = max(emb.shape[0] for emb in new_embs_list)
        padding_embedding = self.input_embeddings(torch.tensor([self.pad_token_id], device=input_embs.device))
        padded_embs = padding_embedding.repeat(batch_size, max_len, 1)

        for i, emb in enumerate(new_embs_list):
            padded_embs[i, :emb.shape[0], :] = emb

        input_embs = padded_embs

        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=input_embs.device)
        for i, label in enumerate(full_labels):
            padded_labels[i, :label.shape[0]] = label

        full_labels = padded_labels

        pad_idx = []
        dev = 1 if self.pad_token_id == self.eos_token_id else 0
        for label in full_labels:
            for k, token in enumerate(label):
                # Mask out retrieval token if it exists.
                if token in [self.pad_token_id, self.start_ret_token_id, self.end_ret_token_id]:
                    label[k+dev:] = -100
                    pad_idx.append(k+dev)
                    break
                if k == len(label) - 1:  # No padding found.
                    pad_idx.append(k + 1)
        assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

        bs, seq_len, embs_dim = input_embs.shape
        if concat_captions:
            assert len(input_embs.shape) == 3, input_embs
            assert len(full_labels.shape) == 2, full_labels
            assert batch_size % 2 == 0
            all_concat_input_embs = []
            all_concat_labels = []

            # Rearrange embeddings and labels (and their padding) to concatenate captions.
            for i in range(batch_size // 2):
                first_idx = i * 2
                second_idx = first_idx + 1
                first_emb = input_embs[first_idx, :pad_idx[first_idx], :]
                first_labels = full_labels[first_idx, :pad_idx[first_idx]]
                first_padding = input_embs[first_idx, pad_idx[first_idx]:, :]
                first_labels_padding = full_labels[first_idx, pad_idx[first_idx]:]

                second_emb = input_embs[second_idx, :pad_idx[second_idx], :]
                second_labels = full_labels[second_idx, :pad_idx[second_idx]]
                second_padding = input_embs[second_idx, pad_idx[second_idx]:, :]
                second_labels_padding = full_labels[second_idx, pad_idx[second_idx]:]

                assert torch.all(first_labels_padding == -100), first_labels_padding
                assert torch.all(second_labels_padding == -100), second_labels_padding
                concat_input_embs = torch.cat([first_emb, second_emb, first_padding, second_padding],
                                              axis=0)  # (T*2, 768)
                concat_labels = torch.cat(
                    [first_labels, second_labels, first_labels_padding, second_labels_padding],
                    axis=0)  # (T*2, 768)
                all_concat_input_embs.append(concat_input_embs)
                all_concat_labels.append(concat_labels)

            # Pad to max length.
            input_embs = torch.stack(all_concat_input_embs, axis=0)  # (N/2, T*2, 768)
            full_labels = torch.stack(all_concat_labels, axis=0)  # (N/2, T*2, 768)
            assert input_embs.shape == (bs // 2, seq_len * 2, embs_dim), input_embs.shape
            assert full_labels.shape == (bs // 2, seq_len * 2), full_labels.shape

        output = self.lm(inputs_embeds=input_embs,
                         labels=full_labels,
                         output_hidden_states=True)


        ce_loss = output.loss
        ce_loss = ce_loss * self.config.cap_loss_scale
        loss = ce_loss

        # if loss is nan, print a bunch of debug info
        if torch.isnan(loss):
            print(f"Loss is NaN. Debug info:")
            print(f"input_embs.shape: {input_embs.shape}")
            print(f"original labels.shape: {labels.shape}")
            print(f"full_labels.shape: {full_labels.shape}")
            print(f"output.loss: {output.loss}")
            print(f"input_ids: ", input_ids)
            print(f"labels: ", labels)
            print(f"full_labels: ", full_labels)
            print(f"input_embs: ", input_embs)
            print("\n\n\n\n")


        return output, full_labels, visual_embs, loss


    def _forward_retrieval(
            self,
            labels: torch.LongTensor,
            input_embs: torch.FloatTensor,
            visual_embs: torch.FloatTensor,
            concat_captions: bool = False,
    ):
        # get the position of the start ret token in the labels
        # last_embedding_idx = (labels == self.ret_token_id).nonzero(as_tuple=True)[1]
        mask = (labels == self.start_ret_token_id)
        flipped_indices = torch.argmax(torch.flip(mask, dims=[1]).int(), dim=1)
        start_last_embedding_idx = labels.shape[1] - 1 - flipped_indices
        start_last_embedding_idx[~mask.any(dim=1)] = -1

        start_ret_tokens = labels[torch.arange(labels.shape[0]), start_last_embedding_idx]
        assert torch.all(start_ret_tokens == self.start_ret_token_id), (start_ret_tokens, self.start_ret_token_id)

        # same for end ret token
        mask = (labels == self.end_ret_token_id)
        flipped_indices = torch.argmax(torch.flip(mask, dims=[1]).int(), dim=1)
        end_last_embedding_idx = labels.shape[1] - 1 - flipped_indices
        end_last_embedding_idx[~mask.any(dim=1)] = -1

        end_ret_tokens = labels[torch.arange(labels.shape[0]), end_last_embedding_idx]
        assert torch.all(end_ret_tokens == self.end_ret_token_id), (start_ret_tokens, self.end_ret_token_id)

        visual_embs = visual_embs.squeeze(2)  # (BATCH_SIZE, NUM_FRAMES, DIM)
        batch_size, num_frames, _ = visual_embs.shape

        # remove the image token from the labels for the retrieval task
        if torch.any(labels == self.img_token_id):
            labels, input_embs, last_embedding_idx = self.remove_img_tokens(labels, input_embs, last_embedding_idx)

        # mask out pad tokens with -100
        full_labels = torch.clone(labels)
        # TODO: if eos == pad we are masking it, think of an alternative where the last eos is visible
        full_labels[full_labels == self.pad_token_id] = -100

        output = self.lm(inputs_embeds=input_embs,
                         labels=full_labels,
                         output_hidden_states=True)

        last_embedding = None
        last_output_logit = None

        ce_loss = output.loss

        ce_loss = ce_loss * self.config.ret_loss_scale
        loss = ce_loss

        start_hidden_states, end_hidden_states = self.ret_text_to_img(output.hidden_states[-1])  # (N, seq_len, dim)

        if num_frames == 1:
            start_relevant_frames = visual_embs
            end_relevant_frames = visual_embs
        else:
            # split the visual embeddings into two parts
            start_relevant_frames = visual_embs[:, :num_frames // 2, :]
            end_relevant_frames = visual_embs[:, num_frames // 2:, :]
            assert start_relevant_frames.shape == end_relevant_frames.shape, (start_relevant_frames.shape, end_relevant_frames.shape)

        for last_embedding_idx, relevant_frames, hidden_state in zip([start_last_embedding_idx, end_last_embedding_idx], [start_relevant_frames, end_relevant_frames], [start_hidden_states, end_hidden_states]):

            if not concat_captions:
                last_embedding = torch.stack(
                    [hidden_state[i, last_embedding_idx[i], :] for i in range(batch_size)], axis=0)  # (N, D)
                last_output_logit = torch.stack(
                    [output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)
            else:
                raise NotImplementedError

            last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)
            # Compute retrieval loss.
            ret_loss = []
            visual_embs_i = torch.mean(relevant_frames, dim=1)  # (N, D)
            visual_embs_i = visual_embs_i / visual_embs_i.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            visual_embs_i = logit_scale * visual_embs_i

            # Compute InfoNCE loss
            caption_loss = info_nce(last_embedding, visual_embs_i, self.negative_queue_image if self.config.use_negatives else None)
            image_loss = info_nce(visual_embs_i, last_embedding, self.negative_queue_text if self.config.use_negatives else None)

            if self.config.use_negatives:
                self.negative_queue_text = self.update_negative_q(self.negative_queue_text, last_embedding)
                self.negative_queue_image = self.update_negative_q(self.negative_queue_image, visual_embs_i)

            ret_loss.append(self.config.ret_loss_scale * (caption_loss + image_loss) / 2.0)

            loss += torch.stack(ret_loss).sum() / len(ret_loss)

        return output, full_labels, last_embedding, last_output_logit, visual_embs, loss

    def update_negative_q(self, queue, new_negatives):
        if queue is None:
            queue = new_negatives
        else:
            queue = torch.cat([queue, new_negatives], dim=0)
            if queue.size(0) > self.max_queue_size:
                queue = queue[-self.max_queue_size:]
        return queue

    def _forward_textgen(
            self,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor] = None,
    ):

        # input_ids = labels.clone()

        # # caption_len here is treated as the source size to mask the source tokens
        # full_labels = labels.clone()
        # for i in range(labels.shape[0]):
        #     full_labels[i, caption_len[i]:] = -100

        output = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = output.loss

        return output, labels, loss

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            mode: str = 'captioning',
            position_ids: Optional[torch.LongTensor] = None,
            concat_captions: bool = False,
            attention_mask: Optional[torch.LongTensor] = None,
    ):

        if self.negative_queue_text is not None:
            self.negative_queue_text = self.negative_queue_text.to(pixel_values.device).detach()
            self.negative_queue_image = self.negative_queue_image.to(pixel_values.device).detach()

        if mode == 'textgen':
            output, full_labels, loss = self._forward_textgen(input_ids, labels, attention_mask)
            return output, full_labels, None, None, None, loss

        if len(pixel_values.shape) > 4: # TODO: normalize this shape
            # the shape is bs, n_frames, c, h, w, so we get the visual embeddings for sample in the batch separately and then we stack them
            visual_embs = []
            for i in range(pixel_values.shape[0]):
                visual_embs.append(self.get_visual_embs(pixel_values[i], mode, None if position_ids is None or mode != "retrieval" else position_ids[i]))
            visual_embs = torch.stack(visual_embs, dim=0)
        else:
            visual_embs = self.get_visual_embs(pixel_values, mode)

        batch_size, vis_seq_len, patches, _ = visual_embs.shape
        if labels is not None:
            assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)

        input_embs = self.input_embeddings(input_ids)  # (N, T, D)

        loss = 0
        ret_loss = 0
        ce_loss = 0

        last_embedding = None
        last_output_logit = None
        hidden_states = []

        if mode == 'captioning':
            output, full_labels, visual_embs, ce_loss = self._forward_captioning(input_ids, labels, input_embs, visual_embs, concat_captions)
            loss = ce_loss
        elif mode == 'retrieval':
            output, full_labels, last_embedding, last_output_logit, visual_embs, loss = self._forward_retrieval(labels, input_embs, visual_embs, concat_captions)
            ret_loss = loss
        else:
            raise NotImplementedError

        return output, full_labels, last_embedding, last_output_logit, visual_embs, loss

    def generate(self, embeddings=torch.FloatTensor, max_len: int = 32,
                 temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
                 ret_scale_factor: float = 1.0, filter_value: float = -float('Inf'), stop_tokens: Optional[List[int]] = []):
        """Runs greedy decoding and returns generated captions.

        Args:
          embeddings: Input condition that the model uses for autoregressive generation.
          max_len: Maximum number of tokens to generate.
          temperature: Used to modulate logit distribution.
          top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
          min_word_tokens: Minimum number of words to generate before allowing a [RET] output.
          ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
          filter_value: Value to assign to tokens that should never be generated.
        Outputs:
          out: (N, T) int32 sequence of output tokens.
          output_embeddings: (N, T, 256) sequence of text output embeddings.
        """
        self.lm.eval()
        if self.eos_token_id not in stop_tokens:
            stop_tokens.extend([self.eos_token_id] if isinstance(self.eos_token_id, int) else self.eos_token_id)

        with torch.no_grad():  # no tracking history
            batch_size, s, _ = embeddings.shape
            # init output with image tokens
            out = None
            past_key_values = None
            output_embeddings = []
            output_logits = []
            cache_position = torch.arange(s, device=embeddings.device).unsqueeze(0).repeat(batch_size, 1)
            # print(f"Handling input with shape {embeddings.shape}")

            for i in range(max_len):
                if 'opt' in self.text_decoder or 'gemma-2' in self.text_decoder:  # gemma 2 has a different cache mechanism that complicates things
                    output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                else:
                    if i == 0:
                        output = self.lm(inputs_embeds=embeddings, use_cache=True, past_key_values=None,
                                         output_hidden_states=True)
                    elif 'gemma-2' in self.text_decoder:
                        # increase the cache position
                        cache_position = torch.cat([cache_position, cache_position[:, -1].unsqueeze(1) + 1], dim=1)
                        print("cache_position.shape", cache_position.shape)
                        output = self.lm(input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values,
                                         cache_position=cache_position, output_hidden_states=True)
                    else:
                        output = self.lm(input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values,
                                         output_hidden_states=True)

                # Collect and sum the hidden states.
                start_hidden_state, end_hidden_state = self.ret_text_to_img(output.hidden_states[-1])  # (N, seq_len, 2048)

                # Add hidden states together.
                start_last_embedding = start_hidden_state / start_hidden_state.norm(dim=-1, keepdim=True)
                end_last_embedding = end_hidden_state / end_hidden_state.norm(dim=-1, keepdim=True)
                output_embeddings.append((start_last_embedding, end_last_embedding))

                logits = output.logits[:, -1, :]  # (N, vocab_size)
                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)

                if self.start_ret_token_id != -1 and self.start_ret_token_id is not None:
                    if i < min_word_tokens:
                        # Eliminate probability of generating [RET] if this is earlier than min_word_tokens.
                        logits[:, self.start_ret_token_id] = filter_value
                    else:
                        # Multiply by scaling factor.
                        logits[:, self.start_ret_token_id] = logits[:, self.start_ret_token_id] * ret_scale_factor
                if self.end_ret_token_id != -1 and self.end_ret_token_id is not None:
                    if i < min_word_tokens:
                        # Eliminate probability of generating [RET] if this is earlier than min_word_tokens.
                        logits[:, self.end_ret_token_id] = filter_value
                    else:
                        # Multiply by scaling factor.
                        logits[:, self.end_ret_token_id] = logits[:, self.end_ret_token_id] * ret_scale_factor

                past_key_values = output.past_key_values

                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / temperature

                    # Apply top-p filtering.
                    if top_p < 1.0:
                        assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value

                    token_weights = logits.exp()  # (N, vocab_size)
                    # remove any nan, inf or < 0 values
                    token_weights[~torch.isfinite(token_weights)] = 0
                    token_weights[token_weights < 0] = 0
                    next_token = torch.multinomial(token_weights, 1)  # (N, 1)

                next_token = next_token.long().to(embeddings.device)
                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                if 'opt' in self.text_decoder or 'gemma-2' in self.text_decoder:
                    next_embedding = self.input_embeddings(next_token)
                    embeddings = torch.cat([embeddings, next_embedding], dim=1)
                elif (self.eos_token_id and (next_token == self.eos_token_id).all()) or next_token in stop_tokens:
                    # End of generation.
                    break

        return out, output_embeddings, output_logits


class VIGIA(PreTrainedModel):
    config_class = VIGIAConfig # Use the custom config
    base_model_prefix = "revistamodel"

    def __init__(self, config: VIGIAConfig): # Takes a config now
        super().__init__(config)
        self.model = VIGIAModel(config)  #Passes the config
        self.config = config

    def __call__(self, pixel_values: Tensor, input_ids: Optional[Tensor] = None, labels: Optional[Tensor] = None,
                 generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
                 ret_scale_factor: float = 1.0, min_word_tokens: int = 0,
                 mode: str = 'captioning', concat_captions: bool = False, stop_tokens: Optional[List[int]] = [],
                 attention_mask: Optional[Tensor] = None, position_ids: Optional[Tensor] = None) -> Tensor:
        if generate:
            return self.model.generate(pixel_values, num_words, temperature=temperature, top_p=top_p, stop_tokens=stop_tokens,
                                       min_word_tokens=min_word_tokens, ret_scale_factor=ret_scale_factor, filter_value=0.0)
        else:
            output = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
                mode=mode,
                concat_captions=concat_captions,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            return output

    def unfreeze_lm(self):
        for param in self.model.lm.parameters():
            param.requires_grad = True
        self.model.config.freeze_lm = False

    def freeze_lm(self):
        for param in self.model.lm.parameters():
            param.requires_grad = False
        self.model.config.freeze_lm = True
    
    def freeze_vm(self):
        for param in self.model.visual_model.parameters():
            param.requires_grad = False
        self.model.config.freeze_vm = True

    def unfreeze_vm(self):
        for param in self.model.visual_model.parameters():
            param.requires_grad = True
        self.model.config.freeze_vm = False

    def freeze_cap(self):
        for param in self.model.visual_embeddings.parameters():
            param.requires_grad = False
        self.model.config.freeze_cap = True

    def unfreeze_cap(self):
        for param in self.model.visual_embeddings.parameters():
            param.requires_grad = True
        self.model.config.freeze_cap = False

    def freeze_ret(self):
        for param in self.model.ret_text_to_img.parameters():
            param.requires_grad = False
        for param in self.model.visual_fc.parameters():
            param.requires_grad = False
        self.model.logit_scale.requires_grad_(False)
        if self.model.config.freeze_lm:
            self.model.input_embeddings.requires_grad_(False)
            if self.model.lm.config.tie_word_embeddings:
                    self.model.output_embeddings.weight.requires_grad_(False)
        self.model.config.freeze_ret = True

    def unfreeze_ret(self):
        for param in self.model.ret_text_to_img.parameters():
            param.requires_grad = True
        for param in self.model.visual_fc.parameters():
            param.requires_grad = True
        self.model.logit_scale.requires_grad_(True)
        if self.model.config.freeze_lm:
            self.model.input_embeddings.requires_grad_(True)
            if self.model.lm.config.tie_word_embeddings:
                    self.model.output_embeddings.weight.requires_grad_(True)
        self.model.config.freeze_ret = False

    def update_freezes(self, freeze_lm = None, freeze_vm = None, freeze_cap = None, freeze_ret = None):
        freeze_lm = freeze_lm if freeze_lm is not None else self.model.config.freeze_lm
        freeze_vm = freeze_vm if freeze_vm is not None else self.model.config.freeze_vm
        freeze_cap = freeze_cap if freeze_cap is not None else self.model.config.freeze_cap
        freeze_ret = freeze_ret if freeze_ret is not None else self.model.config.freeze_ret

        print(f"Updating freezes: LM: {freeze_lm}, VM: {freeze_vm}, CAP: {freeze_cap}, RET: {freeze_ret}")

        if freeze_lm:
            self.freeze_lm()
        else:
            self.unfreeze_lm()

        if freeze_vm:
            self.freeze_vm()
        else:
            self.unfreeze_vm()

        if freeze_cap:
            self.freeze_cap()
        else:
            self.unfreeze_cap()

        if freeze_ret:
            self.freeze_ret()
        else:
            self.unfreeze_ret()