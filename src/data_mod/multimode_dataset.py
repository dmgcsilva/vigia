import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from transformers import AutoFeatureExtractor, AutoProcessor
import transformers
import torch
import json
import jsonlines
import random
from constants import IMG_TOKEN, RET_TOKEN, RET_TOKEN_2, IGNORE_INDEX, USER_CAPTION_REQUEST_TEMPLATES, USER_RETRIEVAL_REQUEST_TEMPLATES, SYSTEM_RETRIEVAL_RESPONSE_TEMPLATES, IMG_END_TOKEN, IMG_START_TOKEN

"""
MAMMOTH STATS (token count)
Max:  40047
Min:  15
Average:  349.68644
75th percentile:  433
80th percentile:  484
90th percentile:  644
95th percentile:  786
99th percentile:  1528
"""


class LazySupervisedDataset(Dataset):
    """
    Used for captioning datasets to load images and captions and build the retrieval conversation.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.debug = debug
        self.data = []

        self.feature_extractor_model = kwargs.get('feature_extractor_model', None)
        self.image_size = kwargs.get('image_size', 224)
        size = {"height": self.image_size, "width": self.image_size}
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_model,
                                                                        do_resize=True,
                                                                        size=size,
                                                                        use_fast=True)
        except:
            self.feature_extractor = AutoProcessor.from_pretrained(self.feature_extractor_model,
                                                                        do_resize=True,
                                                                        size=size,
                                                                        use_fast=True)
        
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'right'
        self.tokenizer.padding_side = 'right'

        # load data from any format and map it to the same format, a list of dictionaries
        if data_path.endswith('.tsv'):
            df = pd.read_csv(data_path, sep="\t")
            assert len(df.columns) == 2, "TSV file must have 2 columns (image, caption)"
            # self.data = self._convert_to_list_of_dicts(df)
            self.data = df
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            assert len(df.columns) == 2, "CSV file must have 2 columns (image, caption)"
            # self.data = self._convert_to_list_of_dicts(df)
            self.data = df
        elif data_path.endswith('.json'):
            with open(data_path, "r") as f:
                self.data = json.load(f)
            assert isinstance(self.data, list), "JSON file must have a list of dictionaries."
        elif data_path.endswith('.jsonl'):
            with jsonlines.open(data_path) as f:
               self.data = [line for line in f]
            assert 'image' in self.data[0], "JSONL file must have an 'image' field in each dictionary."
            assert 'conversations' in self.data[0], "JSONL file must have a 'conversations' field in each dictionary."
        elif data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
            assert len(df.columns) == 2, "Pickle file must have 2 columns (image, caption)"
            # self.data = self._convert_to_list_of_dicts(df)
            self.data = df
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            assert len(df.columns) == 2, "Parquet file must have 2 columns (image, caption)"
            # self.data = self._convert_to_list_of_dicts(df)
            self.data = df
        else:
            raise ValueError("Data format not supported. Supported formats are: TSV, CSV, JSONL, Parquet, Pickle. Received: {}".format(data_path))

        self.raw_targets = ["" for _ in range(len(self.data))]
        self.raw_sources = ["" for _ in range(len(self.data))]

        self.is_test = kwargs.get('is_test', False) or "_val" in data_path or "_test" in data_path or "_eval" in data_path

        self.start_retrieval_token_idx = self.tokenizer.get_vocab().get(RET_TOKEN)
        self.end_retrieval_token_idx = self.tokenizer.get_vocab().get(RET_TOKEN_2)
        self.image_token_idx = self.tokenizer.get_vocab().get(IMG_TOKEN)

        self.max_len = kwargs.get('max_len', self.tokenizer.model_max_length)

        self.context_size = kwargs.get('context_size', 4)
        self.extension_factor = kwargs.get('extension_factor', 1)
        print(f"Got extension factor of size: {self.extension_factor}")
        self.max_samples = kwargs.get('max_samples', len(self.data) * self.extension_factor)

        assert self.start_retrieval_token_idx is not None, f"Token {RET_TOKEN} not found in tokenizer vocab"
        assert self.end_retrieval_token_idx is not None, f"Token {RET_TOKEN_2} not found in tokenizer vocab"
        assert self.image_token_idx is not None, f"Token {IMG_TOKEN} not found in tokenizer vocab"

        print(f"Dataset - start retrieval token idx: {self.start_retrieval_token_idx}, end retrieval token idx: {self.end_retrieval_token_idx}, image token idx: {self.image_token_idx}")
        self._find_longest()
        self.is_first = True

    def _find_longest(self):
        max_length = 0
        self.longest_idx = 0
        for i, data_dict in enumerate(self.data):
            if isinstance(self.data, pd.DataFrame):
                data_dict = self._process_row(self.data.iloc[i])
            conversation = data_dict['conversations']
            full_input, _ = self._build_input_from_conversion(conversation)
            input_length = len(full_input.split())
            if input_length > max_length:
                max_length = input_length
                self.longest_idx = i
        print(f"Longest input has {max_length} words")

    def _build_input_from_conversion(self, conversation):
        """Builds the input from the conversation. And returns the input text with and without the last system response."""
        assert len(conversation) > 1, "The conversation must have at least 2 turns."
        
        assert conversation[-1]['from'] in ['system', 'gpt', 'assistant'], "The last turn in the conversation must be the system's turn."

        # fix 'from' values
        for message in conversation:
            if message['from'] in ['system', 'gpt', 'assistant']:
                message['from'] = 'assistant'
            elif message['from'] in ['human', 'user']:
                message['from'] = 'user'

        is_simple = self.kwargs.get('simple', False)
        if is_simple:
            return " ".join([turn['value'] for turn in conversation]), " ".join([turn['value'] for turn in conversation[:-1]])

        img_token = IMG_START_TOKEN + IMG_TOKEN + IMG_END_TOKEN if not is_simple else IMG_TOKEN

        for msg in conversation:
            msg['value'] = msg['value'].replace('<image>', img_token)
        
        input_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        test_input = self.tokenizer.apply_chat_template(conversation[:-1], tokenize=False)

        return input_text, test_input

    def _process_row(self, row):
        """Converts a single row to the dictionary format."""
        # this is only called for dataframes, which is tipically captioning data thus we add the retrieval and img token
        is_simple = self.kwargs.get('simple', False)
        image_path = row['image_path']
        caption = row['caption']
        img_token = IMG_START_TOKEN + IMG_TOKEN + IMG_END_TOKEN
        conversation = [
            {"from": "user", "value": f"{random.choice(USER_CAPTION_REQUEST_TEMPLATES)}{img_token}"},
            {"from": "assistant", "value": f"{caption}"}
        ]
        return {'image': image_path, 'conversations': conversation}

    def __len__(self):
        return min(len(self.data) * self.extension_factor, self.max_samples)

    def __getitem__(self, idx):
        assert not isinstance(self.data, pd.DataFrame), "LazySupervisedDataset does not support dataframes by itself, please use MultiModeDataset instead."
        idx = idx % len(self.data)
        if self.is_first:
            idx = self.longest_idx # when you stop padding in the dataset this is the best way to avoid finding out that you dont have enough memory mid training
            self.is_first = False
        try:
            # data_dict = self.data[idx]
            data_dict = self.data[idx]
            full_input, test_input = self._build_input_from_conversion(data_dict['conversations'])
            image_path = data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]

            # add to the raw sources and targets (these are used for generation metric calculation at test time)
            self.raw_targets[idx] = full_input
            self.raw_sources[idx] = test_input

            text_diff = full_input.replace(test_input, "")
            input_text = full_input if not self.is_test else test_input

            # Load image
            img = load_images(image_path, self.feature_extractor, self.kwargs.get("image_folder", ""))

            # TODO - apply_chat_template
            tokenized_data = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding='max_length' if not self.is_test else 'do_not_pad',
                    truncation=True,
                    max_length=self.max_len,
                )
            tokens = tokenized_data.input_ids[0]
            labels = tokens.clone()
            attn_mask = tokenized_data.attention_mask[0]

            if not self.is_test:
                labels_length = len(self.tokenizer(text_diff, padding='do_not_pad')['input_ids'])
                # use the position of the first pad token to determine the end of the labels
                labels_end = (labels == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0] if self.tokenizer.pad_token_id in labels else len(tokens)
                # mask the tokens
                labels[:labels_end-labels_length] = IGNORE_INDEX


            caption_len = tokenized_data.attention_mask[0].sum() # for captioning, caption_len is not used but we still return it

            task = 'textgen' if img is None else 'captioning' if IMG_TOKEN in input_text else 'retrieval' if RET_TOKEN in input_text else 'invalid'
            if task == "captioning":
                assert IMG_TOKEN in input_text, f"Image token not found in input text: {input_text}"
            elif task == "retrieval":
                assert RET_TOKEN in input_text, f"Retrieval token not found in input text: {input_text}"
            elif task == 'invalid':
                # print debug info
                print(f"Invalid task for input text: {input_text}")
                print(f"Tokens: {tokens}")
                print(f"Image path: ", image_path)
                print(f"RET_TOKEN in input_text? {RET_TOKEN in input_text}")
                print(f"IMG_TOKEN in input_text? {IMG_TOKEN in input_text}")
                raise ValueError("Could not detrmine the correct task type, see print above for info")

            return_dict = dict(input_ids=tokens, labels=labels, attention_mask=attn_mask, supported_tasks=[task])
            if img is not None:
                return_dict['pixel_values'] = img

            return return_dict
        except Exception as e:
            raise e
            print(f"Error in processing image: {image_path}")
            random_idx = random.randint(0, len(self.data)-1)
            return self.__getitem__(random_idx)


class MultiModeLazyDataset(LazySupervisedDataset):
    
    def __init__(self, tokenizer, data_path, debug=False, **kwargs):
        super().__init__(tokenizer, data_path, debug, **kwargs)

        self.supported_tasks = kwargs.get('supported_tasks', ['captioning', 'retrieval'])

        if len(self.supported_tasks) == 0:
            raise ValueError("At least one supported task must be provided.")
        if any([t not in ['captioning', 'retrieval'] for t in self.supported_tasks]):
            raise ValueError(f"Supported tasks must be either 'captioning' or 'retrieval', got: {self.supported_tasks}")
        if len(self.supported_tasks) == 1:
            if self.supported_tasks[0] == 'captioning':
                self._get_item_func = self._get_captioning_item
            elif self.supported_tasks[0] == 'retrieval':
                self._get_item_func = self._get_retrieval_item
        else:
            self._get_item_func = self._get_multi_mode_item

        self.batch_size = kwargs.get('batch_size', 1)

    def _process_row_captioning(self, row):
        """Converts a single row to the dictionary format."""
        image_path = row['image_path']
        caption = row['caption']
        img_token = IMG_START_TOKEN + IMG_TOKEN + IMG_END_TOKEN
        conversation = [
            {"from": "user", "value": f"{img_token}{random.choice(USER_CAPTION_REQUEST_TEMPLATES)}"},
            {"from": "assistant", "value": f"{caption}"}
        ]
        return {'image': image_path, 'conversations': conversation}

    def _process_row_retrieval(self, row):
        """Converts a single row to the dictionary format."""
        image_path = row['image_path']
        caption = row['caption']
        conversation = [
            {"from": "user", "value": f"{random.choice(USER_RETRIEVAL_REQUEST_TEMPLATES)}{caption}"},
            {"from": "assistant", "value": f"{random.choice(SYSTEM_RETRIEVAL_RESPONSE_TEMPLATES)}{RET_TOKEN}"}
        ]
        return {'image': image_path, 'conversations': conversation}

    def _get_captioning_item(self, idx):
        if self.is_first:
            idx = self.longest_idx # when you stop padding in the dataset this is the best way to avoid finding out that you dont have enough memory mid training
            self.is_first = False
        try:
            data_dict = self._process_row_captioning(self.data.iloc[idx]) if isinstance(self.data, pd.DataFrame) else self.data[idx]
            pixel_values, input_ids, labels, attn_mask, _ = self._shared_item_processing(data_dict, task='captioning')

            assert self.image_token_idx in input_ids, f"Image token ({self.image_token_idx}) not found in tokens: {input_ids}, full_input: {self.tokenizer.decode(input_ids)}"

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attn_mask,
                            supported_tasks=['captioning'])
        except Exception as e:
            # raise e
            print(f"Error in processing cap sample {idx}, with error: {e}")
            random_idx = random.randint(0, len(self.data)-1)
            return self._get_captioning_item(random_idx)

    def _get_retrieval_item(self, idx):
        if self.is_first:
            idx = self.longest_idx # when you stop padding in the dataset this is the best way to avoid finding out that you dont have enough memory mid training
            self.is_first = False
        try:
            data_dict = self._process_row_retrieval(self.data.iloc[idx]) if isinstance(self.data, pd.DataFrame) else self.data[idx]
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
            # raise e
            print(f"Error in processing ret sample {idx}, with error: {e}")
            random_idx = random.randint(0, len(self.data)-1)
            return self._get_retrieval_item(random_idx)

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
            positions = [int(p.split('_')[-1].split('.')[0]) // 20 for p in paths]
            position_ids = torch.tensor(positions, dtype=torch.long)
        elif task == 'retrieval':
            position_ids = torch.tensor([0]*len(data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]), dtype=torch.long)

        return imgs, input_ids, labels, attn_mask, position_ids

    def _get_multi_mode_item(self, idx):
        # get the correct dataset, knowding that the batches are interleaved between the caption and retrieval datasets
        batch_idx = idx // self.batch_size
        if batch_idx % 2 == 0:  # even index, return caption dataset
            # print(f"Idx {idx} - Batch idx {batch_idx} - Caption dataset")
            item = self._get_captioning_item(idx - ((batch_idx - 1) * self.batch_size))
        else: # odd index, return retrieval dataset
            # print(f"Idx {idx} - Batch idx {batch_idx} - Retrieval dataset")
            item = self._get_retrieval_item(idx - ((batch_idx - 1) * self.batch_size))

        for k, v in item.items():
            if k in ['input_ids', 'labels', 'attention_mask', 'position_ids']:
                item[k] = v.to(torch.long)
            if k in ['pixel_values']:
                item[k] = v.to(torch.float32)
        
        return item

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        return self._get_item_func(idx)

    def __len__(self):
        return super().__len__() * len(self.supported_tasks)
    


def load_images(images, feature_extractor, base_folder):
    """
    Load images from a list of paths and return the pixel values.
    """
    if not isinstance(images, list):
        images = [images]
    pixel_values = []
    for image in images:
        img = Image.open(os.path.join(base_folder, image))
        img = feature_extractor(images=img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]
        pixel_values.append(img)

    pixel_values = torch.stack(pixel_values)

    return pixel_values