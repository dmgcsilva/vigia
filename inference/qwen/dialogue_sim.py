import json
import os

import torch
from tqdm import tqdm
from .. import inference_utils
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..model import IMG_TOKEN
from time import time
from ..dataset import load_images

FILE_PATH = "/home/dmgcsilva/project/DATA/mmplanllm/dialogues_with_vqa_test.json"


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the checkpoint file")
    parser.add_argument("--max-dialogues", type=int, default=100, help="Maximum number of dialogues to process")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")


    return parser.parse_args()


def load_model_and_tokenizer(args):

    print(f"Loading model and tokenizer from {args.ckpt_path}")
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckpt_path, torch_dtype=torch.float16)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.ckpt_path)

    # model in BF16
    model = model.to(dtype=torch.bfloat16)
    # model in eval mode
    model.eval()

    return model, processor 

def generate_response(model, tokenizer, task, history, images):
    item = build_inputs(tokenizer, task, history, images)

    # move inputs to device
    device = next(model.parameters()).device
    for k in item:
        if isinstance(item[k], torch.Tensor):
            item[k] = item[k].to(device)
        elif isinstance(item[k], dict):
            for kk in item[k]:
                if isinstance(item[k][kk], torch.Tensor):
                    item[k][kk] = item[k][kk].to(device)

    generate_ids = model.generate(**item, max_new_tokens=128, do_sample=False)
    gen_step = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()

    del item
    torch.cuda.empty_cache()

    return gen_step


def build_inputs(tokenizer, task, history, images):
    first_turn = "You are a helpful AI conversational assistant. You can retrieve images or video moments by generating the [RET] token.\nRight now you are helping a user through the task below. Guiden them step by step and answer any questions they may have.\n\n" + task
    convs = [{'from': 'assistant', 'value': first_turn}] + history

    data_dict = {
        'conversations': convs,
        'image': images,
    }

    converted_messages = []
    img_count = 0
    for msg in data_dict['conversations']:
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
                    {"type": "text", "text": msg['value']},
                ],
            })
        else:
            raise ValueError(f"Unknown role {msg['from']} in message {msg}")

    inputs = tokenizer.apply_chat_template(
        converted_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    return inputs

def get_task_from_dialogue(data):
    """
    Extracts the recipe title and steps from a task dictionary 
    and returns a formatted string.
    """
    if 'recipe' in data['task']:
        recipe = data.get('task', {}).get('recipe', {})
    elif 'task' in data['task']:
        recipe = data.get('task', {}).get('task', {})
    else:
        raise ValueError(f"No recipe found in task data: {data['task']}") 
    
    title = recipe.get('displayName', 'Untitled Recipe')
    instructions = recipe.get('instructions', [])
    
    output_lines = ["Task: " + title]
    for index, step in enumerate(instructions, 1):
        step_text = step.get('stepText', '').strip()
        output_lines.append(f"Step {index}: {step_text}")
    
    return "\n".join(output_lines)

def clean_path(path):
    path = path.replace('/data/dmgc.silva/datasets/tasty_dataset/ALL_RECIPES_without_videos/', "/home/dmgcsilva/project/DATA/mmplanllm/images/tasty/")
    path = path.replace('/storagebk/datasets/COIN/', '/home/dmgcsilva/project/DATA/mmplanllm/images/')
    path = path.replace('frames_reduced_reduced_reduced', 'frames_reduced')
    path = path.replace('frames_reduced_reduced', 'frames_reduced')
    return path

if __name__ == "__main__":
    args = get_args()
    model, tokenizer = load_model_and_tokenizer(args)

    dialogues = []
    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    # for now we exlucde turns with VisualMomentRetrievalIntent becasue I don't know how they'd be evaluated
    for dialogue in data.keys():
        if any(['ARTIFICIAL.VisualMomentRetrievalIntent' == turn['intent'] for turn in data[dialogue]['dialog']]):
            continue
        dialogues.append(data[dialogue])

    generated_dialogues = []
    with torch.inference_mode():

        for dialogue in tqdm(dialogues):
            task = get_task_from_dialogue(dialogue)
            history = []
            images = []
            for turn in dialogue['dialog']:
                user_utterance = turn['user']
                if turn.get('relevant_image', None) is not None:
                    images.append(clean_path(turn['relevant_image']))
                    user_utterance = "<image>\n" + user_utterance
                history.append({'from': 'user', 'value': user_utterance})
                response = generate_response(model, tokenizer, task, history, images)

                history.append({'from': 'assistant', 'value': response})
            generated_dialogues.append({'image': images, 'conversations': history, 'task': dialogue['task']})

    out_path = "/home/dmgcsilva/workbench/followup/code/inference/results/qwen25_gen_dialogues.json"
    with open(out_path, "w") as f:
        json.dump(generated_dialogues, f, indent=4)

    print(f"Generated dialogues saved to {out_path}")
# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m inference.qwen.dialogue_sim --device_id 0