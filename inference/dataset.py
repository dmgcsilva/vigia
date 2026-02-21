import json
import os
import random
import time

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoFeatureExtractor, AutoProcessor

SEP_TOKEN = "\n"
IMG_TOKEN = "[IMG]"
IMG_START_TOKEN = "<im_start>"
IMG_END_TOKEN = "<im_end>"
RET_TOKEN = "[RET]"
RET_TOKEN_2 = "[RET2]"

IGNORE_INDEX = -100


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


class MMPlanLLMDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.debug = debug
        self.data = []

        self.feature_extractor_model = kwargs.get('feature_extractor_model', None)
        size = {"height": 224, "width": 224}
        self.is_llava = kwargs.get('is_llava', False)
        self.is_idefics2 = kwargs.get('is_idefics2', False)
        self.feature_extractor = None
        if not self.is_llava and not self.is_idefics2:
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
        if data_path.endswith('.json'):
            with open(data_path, "r") as f:
                self.data = json.load(f)
            assert isinstance(self.data, list), "JSON file must have a list of dictionaries."
        else:
            raise ValueError("Data format not supported. Supported formats are: JSON. Received: {}".format(data_path))

        self.raw_targets = ["" for _ in range(len(self.data))]
        self.raw_sources = ["" for _ in range(len(self.data))]

        self.is_test = kwargs.get('is_test', False) or "_val" in data_path or "_test" in data_path or "_eval" in data_path

        self.start_retrieval_token_idx = self.tokenizer.get_vocab().get(RET_TOKEN) if not self.is_llava and not self.is_idefics2 else None
        self.end_retrieval_token_idx = self.tokenizer.get_vocab().get(RET_TOKEN_2) if not self.is_llava and not self.is_idefics2 else None
        self.image_token_idx = self.tokenizer.get_vocab().get(IMG_TOKEN) if not self.is_llava and not self.is_idefics2 else None

        self.max_len = kwargs.get('max_len', self.tokenizer.model_max_length if not self.is_llava and not self.is_idefics2 else 2048)

        self.context_size = kwargs.get('context_size', 4)
        self.extension_factor = kwargs.get('extension_factor', 1)
        print(f"Got extension factor of size: {self.extension_factor}")
        self.max_samples = kwargs.get('max_samples', len(self.data) * self.extension_factor)

        assert self.is_llava or self.is_idefics2 or self.start_retrieval_token_idx is not None, f"Token {RET_TOKEN} not found in tokenizer vocab"
        assert self.is_llava or self.is_idefics2 or self.end_retrieval_token_idx is not None, f"Token {RET_TOKEN_2} not found in tokenizer vocab"
        assert self.is_llava or self.is_idefics2 or self.image_token_idx is not None, f"Token {IMG_TOKEN} not found in tokenizer vocab"

        print(f"Dataset - start retrieval token idx: {self.start_retrieval_token_idx}, end retrieval token idx: {self.end_retrieval_token_idx}, image token idx: {self.image_token_idx}")

        self.modes = []
        self._get_modes()
        self.vis_embs_cache = {}
    
    def _get_modes(self):
        # TODO: Unravel this into:
        # 1. Text generation (text only)
        # 2. MM Text Geenration (image + text input, text output)
        # 3. Visually Informed Step Generation
        # 4. VQA
        # 5. Retrieval
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

    def _build_input_from_conversion(self, conversation):
        """Builds the input from the conversation. And returns the input text with and without the last system response."""

        # fix 'from' values
        for message in conversation:
            if message['from'] in ['system', 'gpt', 'assistant']:
                message['from'] = 'assistant'
            elif message['from'] in ['human', 'user']:
                message['from'] = 'user'

        img_token = IMG_START_TOKEN + IMG_TOKEN + IMG_END_TOKEN

        for msg in conversation:
            msg['value'] = msg['value'].replace('<image>', img_token)
        
        input_text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        return input_text

    def __getitem_textgen(self, idx):
        try:
            _, input_ids, labels, attn_mask, _ = self._shared_item_processing(idx, task='textgen')
            pixel_values = torch.zeros(1, 1, 1, 1) # dummy value
            return dict(input_ids=input_ids, labels=labels, attention_mask=attn_mask, supported_tasks=['textgen'], pixel_values=pixel_values)
        except Exception as e:
            raise e

    def __getitem_cap(self, idx):
        try:
            pixel_values, input_ids, labels, attn_mask, _ = self._shared_item_processing(idx, task='captioning')

            assert self.image_token_idx in input_ids, f"Image token ({self.image_token_idx}) not found in tokens: {input_ids}, full_input: {self.tokenizer.decode(input_ids)}"

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attn_mask,
                            supported_tasks=['captioning'])
        except Exception as e:
            raise e

    def __getitem_ret(self, idx):
        try:
            pixel_values, input_ids, labels, attn_mask, position_ids = self._shared_item_processing(idx, task='retrieval')

            assert self.start_retrieval_token_idx in input_ids, f"Missing retrieval token {self.start_retrieval_token_idx} in tokens {input_ids}"
            assert self.end_retrieval_token_idx in input_ids, f"Missing retrieval token {self.end_retrieval_token_idx} in tokens {input_ids}"

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, attention_mask=attn_mask, position_ids=position_ids,
                            supported_tasks=['retrieval'])
        except Exception as e:
            raise e

    def _shared_item_processing(self, idx, task):
        data_dict = self.data[idx]
        input_text = self._build_input_from_conversion(data_dict['conversations'][:-1])
        target_text = data_dict['conversations'][-1]['value']

        self.raw_sources[idx] = input_text
        self.raw_targets[idx] = target_text

        if task == 'retrieval':
            input_text += target_text

        imgs = load_images(data_dict['image'], self.feature_extractor, self.kwargs.get("image_folder", ""))

        tknd = self.tokenizer(input_text, return_tensors="pt", padding='do_not_pad', truncation=False, max_length=self.tokenizer.model_max_length)
        input_ids = tknd['input_ids'].squeeze()
        attn_mask = tknd['attention_mask'].squeeze()

        position_ids = None
        if task == 'retrieval' and ((isinstance(data_dict['image'], str) and 'frame_' in data_dict['image']) or (isinstance(data_dict['image'], list) and all(['frame_' in i for i in data_dict['image']]))):
            paths = data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]
            positions = [int(p.split('_')[-1].split('.')[0]) // 20 for p in paths]
            position_ids = torch.tensor(positions, dtype=torch.long)
        elif task == 'retrieval':
            position_ids = torch.tensor([0]*len(data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]), dtype=torch.long)

        return imgs, input_ids, None, attn_mask, position_ids
    
    def __getitem_llava(self, idx):
        data_dict = self.data[idx]
        converted_messages = []
        img_count = 0
        for msg in data_dict['conversations'][:-1]:
            if msg['from'] == 'user':
                converted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg['value'].replace('<image>\n', '')},
                    ],
                })
                if '<image>' in msg['value'] or IMG_TOKEN in msg['value']:
                    converted_messages[-1]['content'] = [
                        {"type": "image", "url": data_dict['image'][img_count] if isinstance(data_dict['image'], list) else data_dict['image']},
                    ] + converted_messages[-1]['content']
                    img_count += 1
            elif msg['from'] == 'assistant':
                converted_messages.append({
                    "role": "assistant",
                    "content": [
                        # {"type": "text", "text": msg['value'].replace("Guiden them step by step and answer any questions they may have.", "Guide them step by step and answer any questions they may have. DO NOT be proactive, let the user lead the conversation, and be factual with respect to the task, do not add exttra commentary unless asked. NO Markdown please.")},
                        {"type": "text", "text": msg['value']},
                    ],
                })
            else:
                raise ValueError(f"Unknown role {msg['from']} in message {msg}")

        self.raw_targets[idx] = data_dict['conversations'][-1]['value']

        inputs = self.tokenizer.apply_chat_template(
            converted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        self.raw_sources[idx] = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        return inputs

    def __getitem_idefics2(self, idx):
        data_dict = self.data[idx]
        converted_messages = []
        images = None
        if data_dict['image'] is not None and data_dict['image'] != []:
            images = [Image.open(img) for img in data_dict['image']] if isinstance(data_dict['image'], list) else [Image.open(data_dict['image'])]

        for msg in data_dict['conversations'][:-1]:
            if msg['from'] == 'user':
                converted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg['value'].replace('<image>\n', '')},
                    ],
                })
                if '<image>' in msg['value'] or IMG_TOKEN in msg['value']:
                    converted_messages[-1]['content'].append({'type': 'image'})
            elif msg['from'] == 'assistant':
                converted_messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": msg['value']},
                    ],
                })
            else:
                raise ValueError(f"Unknown role {msg['from']} in message {msg}")

        self.raw_targets[idx] = data_dict['conversations'][-1]['value']

        text = self.tokenizer.apply_chat_template(converted_messages, add_generation_prompt=True)

        inputs = self.tokenizer(images=images, text=text, return_tensors="pt")

        self.raw_sources[idx] = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        return inputs

    def __getitem__(self, idx):
        mode = self.modes[idx]

        if self.is_llava:
            assert mode != "ret", "Retrieval mode is not supported for Llava"
            return self.__getitem_llava(idx)
        if self.is_idefics2:
            assert mode != "ret", "Retrieval mode is not supported for Idefics2"
            return self.__getitem_idefics2(idx)

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


    def get_target_embeddings(self, idx, model):
        # Given an idx it return the visual embeddings for every frame that belongs to the same video

        # get video directory (the folder in which the frames are stored)
        img_path = self.data[idx]['image']
        if img_path is None:
            raise ValueError(f"No image path found for item at index {idx}")

        if isinstance(img_path, str):
            video_dir = os.path.dirname(img_path)
        elif isinstance(img_path, list):
            assert all([os.path.dirname(img_path[0]) == os.path.dirname(img_path[i]) for i in range(len(img_path))]), f"All images must be from the same video, but got {img_path}"
            video_dir = os.path.dirname(img_path[0])

        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
        # sort the frames so that the frame at 0 is the first frame in the video (000000.jpg)
        frame_files.sort()

        recipe_name = video_dir.split('/')[-2]

        target_ids = [int(img_path[0].split('/')[-1].replace('.jpg', '').replace('frame_', '')), int(img_path[-1].split('/')[-1].replace('.jpg', '').replace('frame_', ''))]
        # print(f"got target ids: {target_ids} for recipe {recipe_name}")

        if recipe_name in self.vis_embs_cache:
            return target_ids, self.vis_embs_cache[recipe_name]

        if hasattr(model, "model"):
            device = model.model.lm.device
        else: # is clip model
            device = model.device

        start_time = time.time()

        pixel_values = []
        positions = [int(p.split('_')[-1].split('.')[0]) for p in frame_files]
        for img in frame_files:
            try:
                pth = os.path.join(video_dir, img)
                img_object = Image.open(pth).convert('RGB')
                images = self.feature_extractor(images=img_object, return_tensors="pt").pixel_values[0, ...].to(device)
                pixel_values.append(images)
            except Exception as e:
                # print(f'Error reading {img}: {e}, continuing...')
                # continue
                raise e

        pixel_values = torch.stack(pixel_values, dim=0)
        position_ids = torch.tensor(positions, dtype=torch.long, device=device)


        with torch.no_grad():
            # check if there is a model inside model
            if hasattr(model, "model"):
                vis_embs = model.model.get_visual_embs(pixel_values, mode="retrieval", position_offset=position_ids).squeeze(1)
            else: # is a clip model
                vis_embs = model.get_image_features(pixel_values).to(device)

        self.vis_embs_cache[recipe_name] = vis_embs

        return target_ids, vis_embs