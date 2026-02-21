import json
import os
import random
import time

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from src.data_mod.multimode_dataset import LazySupervisedDataset
from constants import IGNORE_INDEX, IMG_TOKEN


def load_images(images, feature_extractor, base_folder):
    """
    Load images from a list of paths and return the pixel values.
    """
    if images is None or (isinstance(images, list) and len(images) == 0):
        return None
    if not isinstance(images, list):
        images = [images]
    pixel_values = []
    for image in images:
        img = Image.open(os.path.join(base_folder, image))
        img = feature_extractor(images=img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]
        pixel_values.append(img)

    pixel_values = torch.stack(pixel_values)

    return pixel_values


class MMPlanLLMDataset(LazySupervisedDataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        super(MMPlanLLMDataset, self).__init__(tokenizer, data_path, debug, **kwargs)

        print("max len", self.max_len)
        self.modes = []
        self._get_modes()
        self.is_first = True
        self._get_longest()


    def _get_longest(self):
        longest = {'ret': 0, 'cap': 0, 'textgen': 0}
        longest_idx = {'ret': 0, 'cap': 0, 'textgen': 0}
        for i, item in enumerate(self.data):
            mode = self.modes[i]
            full_input, _ = self._build_input_from_conversion(item['conversations'])
            word_count = len(full_input.split())
            img_count = len(item['image']) if isinstance(item['image'], list) else 1 if item['image'] is not None else 0
            if mode != 'ret':
                img_count = 0
            item_len = word_count + img_count * 256
            if item_len > longest[mode]:
                longest[mode] = item_len
                longest_idx[mode] = i
        print(f"Longest items: {longest}")
        self.longest = longest_idx

    
    def _get_modes(self):
        counts = {'ret': 0, 'cap': 0, 'textgen': 0}
        for item in self.data:
            if any(['[RET][RET2]' in m['value'] for m in item['conversations'][1:]]):
                self.modes.append('ret')
                counts['ret'] += 1
            elif any(['<image>' in m['value'] or IMG_TOKEN in m['value'] for m in item['conversations'][1:]]):
                self.modes.append('cap')
                counts['cap'] += 1
            else:
                self.modes.append('textgen')
                counts['textgen'] += 1
        
        print(f"Modes: {counts}")

    def __getitem_textgen(self, idx):
        try:
            data_dict = self.data[idx]
            _, input_ids, labels, attn_mask, _ = self._shared_item_processing(data_dict, task='textgen')
            pixel_values = torch.zeros(1, 1, 1, 1) # dummy value
            return dict(input_ids=input_ids, labels=labels, attention_mask=attn_mask, supported_tasks=['textgen'], pixel_values=pixel_values)
        except Exception as e:
            #raise e
            print(f"Error in processing textgen sample {idx}, with error: {e}")
            random_idx = random.randint(0, len(self.data)-1)
            return self.__getitem_textgen(random_idx)

    def __getitem_cap(self, idx):
        if self.is_first:
            idx = self.longest['cap']
        try:
            data_dict = self.data[idx]
            pixel_values, input_ids, labels, attn_mask, _ = self._shared_item_processing(data_dict, task='captioning')

#            assert self.image_token_idx in input_ids, f"Image token ({self.image_token_idx}) not found in tokens: {input_ids}, full_input: {self.tokenizer.decode(input_ids)}"

            if self.is_first:
                img_count = len(data_dict['image']) if isinstance(data_dict['image'], list) else 1 if data_dict['image'] is not None else 0
                print(f"Longest sample, token count: {len(input_ids)}, image count: {img_count}({img_count*256} tokens), for a total of {len(input_ids) + img_count*256} tokens")
                self.is_first = False

            assert self.image_token_idx in input_ids, f"Image token ({self.image_token_idx}) not in inputs ids {input_ids}, full-input {self.tokenizer.decode(input_ids)}"

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attn_mask,
                            supported_tasks=['captioning'])
        except Exception as e:
            #raise e
            print(f"Error in processing cap sample {idx}, with error: {e}")
            random_idx = random.randint(0, len(self.data)-1)
            return self.__getitem_cap(random_idx)

    def __getitem_ret(self, idx):
        try:
            data_dict = self.data[idx]
            pixel_values, input_ids, labels, attn_mask, position_ids = self._shared_item_processing(data_dict, task='retrieval')

            if self.is_test:
                # for test mode, we do not need to add the RET or EOS tokens as they should be generated by the model
                assert all([t != self.start_retrieval_token_idx and t != self.end_retrieval_token_idx for t in input_ids])
            elif self.start_retrieval_token_idx not in input_ids and self.end_retrieval_token_idx not in input_ids:
                print(f"Retrieval token not found in input_ids, adding it manually")
                input_ids[-1] = self.end_retrieval_token_idx
                input_ids[-2] = self.start_retrieval_token_idx
                labels[-1] = self.end_retrieval_token_idx
                labels[-2] = self.start_retrieval_token_idx
                attn_mask[-1] = 1
                attn_mask[-2] = 1
            elif self.start_retrieval_token_idx not in labels or self.end_retrieval_token_idx not in labels:
                print(f"Retrieval token not found in labels, adding it manually")
                if self.start_retrieval_token_idx not in labels:
                    start_ret_pos = (input_ids == self.start_retrieval_token_idx).nonzero(as_tuple=True)[0][-1]
                    labels[start_ret_pos] = self.start_retrieval_token_idx
                    attn_mask[start_ret_pos] = 1
                elif self.end_retrieval_token_idx not in labels:
                    end_ret_pos = (input_ids == self.end_retrieval_token_idx).nonzero(as_tuple=True)[0][-1]
                    labels[end_ret_pos] = self.end_retrieval_token
                    attn_mask[end_ret_pos] = 1

            assert self.start_retrieval_token_idx in input_ids, f"Missing retrieval token {self.start_retrieval_token_idx} in tokens {input_ids}"
            assert self.start_retrieval_token_idx in labels, f"Missing retrieval token {self.start_retrieval_token_idx} in lables {labels} \n tokens {input_ids}"
            assert self.end_retrieval_token_idx in input_ids, f"Missing retrieval token {self.end_retrieval_token_idx} in tokens {input_ids}"
            assert self.end_retrieval_token_idx in labels, f"Missing retrieval token {self.end_retrieval_token_idx} in lables {labels} \n tokens {input_ids}"

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attn_mask, position_ids=position_ids,
                            supported_tasks=['retrieval'])
        except Exception as e:
            full_input, test_input = self._build_input_from_conversion(data_dict['conversations'])
            print(f"Full input: {full_input}")
            print(f"Test input: {test_input}")
            print("input_ids size:", input_ids.shape)
            print("labels size:", labels.shape)
           # raise e
            print(f"Error in processing ret sample {idx}, with error: {e}")
            random_idx = random.randint(0, len(self.data)-1)
            return self.__getitem_ret(random_idx)

    def _shared_item_processing(self, data_dict, task):
        full_input, test_input = self._build_input_from_conversion(data_dict['conversations'])

        imgs = load_images(data_dict['image'], self.feature_extractor, self.kwargs.get("image_folder", ""))

        test_tokenized = self.tokenizer(test_input, return_tensors="pt", padding='do_not_pad', truncation=False, max_length=self.tokenizer.model_max_length)['input_ids'][0]
        test_ids_len = test_tokenized.size(0)
        full_tokenized = self.tokenizer(full_input, return_tensors="pt", padding='do_not_pad', truncation=False, max_length=self.tokenizer.model_max_length)
        input_ids = full_tokenized['input_ids'][0]
        attn_mask = full_tokenized['attention_mask'][0]
        labels = input_ids.clone()

        labels[:test_ids_len] = IGNORE_INDEX

        if input_ids.size(0) > self.max_len:
            if task == 'captioning': # for captioning we truncate on the right to preserve the image token
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]
                attn_mask = attn_mask[:self.max_len]
            else: # for retrieval we truncate on the left to preserve the retrieval token
                input_ids = input_ids[-self.max_len:]
                labels = labels[-self.max_len:]
                attn_mask = attn_mask[-self.max_len:]

        position_ids = None
        if task == 'retrieval' and ((isinstance(data_dict['image'], str) and 'frame_' in data_dict['image']) or (isinstance(data_dict['image'], list) and all(['frame_' in i for i in data_dict['image']]))):
            paths = data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]
            positions = [int(p.split('_')[-1].split('.')[0]) for p in paths]
            position_ids = torch.tensor(positions, dtype=torch.long)
        elif task == 'retrieval':
            position_ids = torch.tensor([0]*len(data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]), dtype=torch.long)

        return imgs, input_ids, labels, attn_mask, position_ids

    def __getitem__(self, idx):
        mode = self.modes[idx]
        if mode == 'ret':
            return self.__getitem_ret(idx)
        elif mode == 'cap':
            return self.__getitem_cap(idx)
        elif mode == 'textgen':
            return self.__getitem_textgen(idx)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def __len__(self):
        return len(self.data)




class MMPlanLLMModeSampler(torch.utils.data.Sampler):

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.modes_types = list(set([m[0] for m in dataset.modes]))  # List[str]
        self.order = []
        self.batch_size = batch_size

        print(f"Modes: {self.modes_types}")

        self._build_order_list()


    def _build_order_list(self):
        # iterate through the dataset and build the order list so that each batch has only one mode
        mode_lists = {mode: [] for mode in self.modes_types}
        for i in range(len(self.dataset)):
            mode = self.dataset.modes[i]
            mode_lists[mode[0]].append(i)

        print(f"Total valid items b4 prune: {sum([len(mode_lists[mode]) for mode in mode_lists])}")

        # remove the end of the lists so that they are divisible by the batch size
        for mode in mode_lists:
            old_size = len(mode_lists[mode])
            new_size = (old_size // self.batch_size) * self.batch_size
            mode_lists[mode] = mode_lists[mode][:new_size]
            print(f"Mode {mode} pruned from {old_size} to {new_size}")

        print(f"Total valid items: {sum([len(mode_lists[mode]) for mode in mode_lists])}")

        # Calculate total batches for each mode
        total_batches = {mode: len(mode_lists[mode]) // self.batch_size for mode in mode_lists}

        # Determine how often to insert batches of each mode
        min_batches = min(total_batches.values())
        interleave_intervals = {mode: total_batches[mode] // min_batches for mode in mode_lists}

        print(f"Interleave intervals: {interleave_intervals}")

        # Initialize counters and order list
        self.order = []
        mode_batch_counters = {mode: 0 for mode in mode_lists}

        # Interleave batches from different modes
        while any(mode_batch_counters[mode] < total_batches[mode] for mode in mode_lists):
            for mode in sorted(mode_lists, key=lambda x: interleave_intervals[x]):
                if mode_batch_counters[mode] < total_batches[mode]:
                    start_index = mode_batch_counters[mode] * self.batch_size
                    end_index = start_index + (self.batch_size * interleave_intervals[mode])
                    self.order.extend(mode_lists[mode][start_index:min(end_index, len(mode_lists[mode]))])  # Add interleave_interval batches of this mode

                    # Advance the insertion point by the interleave interval times batch size
                    mode_batch_counters[mode] += interleave_intervals[mode]
                    if mode_batch_counters[mode] >= total_batches[mode]:
                        # If we've scheduled all batches for this mode, stop trying to add more
                        mode_batch_counters[mode] = total_batches[mode]

        print(f"Total items in order list: {len(self.order)}")

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, item):
        return self.dataset[item]
