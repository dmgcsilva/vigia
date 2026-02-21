import json
import os

import torch
from tqdm import tqdm
from .. import inference_utils
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor

from ..model import IMG_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN, RET_TOKEN, RET_TOKEN_2, VIGIA
from time import time
from ..dataset import load_images

FILE_PATH = "/home/dmgcsilva/project/DATA/mmplanllm/dialogues_with_vqa_test.json"


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--max-dialogues", type=int, default=100, help="Maximum number of dialogues to process")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")


    return parser.parse_args()


def load_model_and_tokenizer(args):

    print(f"Loading model and tokenizer from {args.ckpt_path}")

    args.ckpt_path = args.ckpt_path[:-1] if args.ckpt_path[-1] == '/' else args.ckpt_path

    model = VIGIA.from_pretrained(args.ckpt_path, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True, trust_remote_code=True)
    assert tokenizer.get_vocab().get(RET_TOKEN) == model.model.start_ret_token_id, f"Retrieval token mismatch: {tokenizer.get_vocab().get(RET_TOKEN)} != {model.model.start_ret_token_id}"
    assert tokenizer.get_vocab().get(RET_TOKEN_2) == model.model.end_ret_token_id, f"Retrieval token mismatch: {tokenizer.get_vocab().get(RET_TOKEN_2)} != {model.model.end_ret_token_id}"
    assert tokenizer.get_vocab().get(IMG_TOKEN) == model.model.img_token_id, f"Image token mismatch: {tokenizer.get_vocab().get(IMG_TOKEN)} != {model.model.img_token_id}"


    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # model in BF16
    model = model.to(dtype=torch.bfloat16)
    # model in eval mode
    model.eval()

    tokenizer.model_max_length = 2048

    size = {"height": 224, "width": 224}
    feature_extractor = None
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model.config.visual_encoder,
                                                                    do_resize=True,
                                                                    size=size,
                                                                    use_fast=True)
    except:
        feature_extractor = AutoProcessor.from_pretrained(model.config.visual_encoder,
                                                                    do_resize=True,
                                                                    size=size,
                                                                    use_fast=True)
    
    tokenizer.truncation_side = 'right'
    tokenizer.padding_side = 'right'

    return model, tokenizer, feature_extractor

def generate_response(model, tokenizer, feature_extractor, prompt, images, stop_tokens):
    item = build_inputs(tokenizer, prompt, images, feature_extractor)

    supported_tasks = item.pop('supported_tasks')
    if 'captioning' in supported_tasks:
        task = 'captioning'
    else:
        task = 'textgen'

    if task == 'captioning':
        text_embeddings = model.model.input_embeddings(item['input_ids'].to(device=model.model.lm.device).unsqueeze(0)).squeeze(0) # S, D
        visual_embeddings = model.model.get_visual_embs(item['pixel_values'].to(device=model.model.lm.device), mode='captioning') # N, D
        # get all postions of the model.model.img_token_id
        img_token_positions = (item['input_ids'] == model.model.img_token_id).nonzero(as_tuple=True)[0]
        assert len(img_token_positions) == visual_embeddings.shape[0], f"Image token positions {len(img_token_positions)} do not match visual embeddings {visual_embeddings.shape[0]}"
        final_embs = []
        last_pos = 0
        for i, pos in enumerate(img_token_positions):
            final_embs.append(text_embeddings[last_pos:pos])
            final_embs.append(visual_embeddings[i])
            last_pos = pos + 1
        final_embs.append(text_embeddings[last_pos:])
        input_embs = torch.cat(final_embs, dim=0).unsqueeze(0).to(device=model.model.lm.device) # 1, S, D

        outputs, _, output_logits = model(
            input_embs,
            None,
            None,
            generate=True,
            num_words=128,
            stop_tokens=stop_tokens,
            temperature=0.0,
        )

        # get the caption
        gen_step = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if '<|im_end|>' in gen_step:
            gen_step = gen_step.split('<|im_end|>')[0]

    else:
        text_embeddings = model.model.input_embeddings(item['input_ids'].to(device=model.model.lm.device).unsqueeze(0))

        outputs, _, output_logits = model(
            text_embeddings,
            None,
            None,
            generate=True,
            num_words=128,
            temperature=0.0,
            stop_tokens=stop_tokens,
        )

        # get the caption
        gen_step = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if '<|im_end|>' in gen_step:
            gen_step = gen_step.split('<|im_end|>')[0]


    return gen_step


def build_inputs(tokenizer, input_text, images, feature_extractor):
    task = 'captioning'
    if task == 'retrieval':
        input_text += target_text

    pixel_values = load_images(images, feature_extractor, "") if images is not None and len(images) > 0 else torch.zeros(1, 1, 1, 1) # dummy value

    tknd = tokenizer(input_text, return_tensors="pt", padding='do_not_pad', truncation=False, max_length=tokenizer.model_max_length)
    input_ids = tknd['input_ids'].squeeze()
    attn_mask = tknd['attention_mask'].squeeze()

    position_ids = None
    if task == 'retrieval' and ((isinstance(data_dict['image'], str) and 'frame_' in data_dict['image']) or (isinstance(data_dict['image'], list) and all(['frame_' in i for i in data_dict['image']]))):
        paths = data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]
        positions = [int(p.split('_')[-1].split('.')[0]) // 20 for p in paths]
        position_ids = torch.tensor(positions, dtype=torch.long)
    elif task == 'retrieval':
        position_ids = torch.tensor([0]*len(data_dict['image'] if isinstance(data_dict['image'], list) else [data_dict['image']]), dtype=torch.long)

    return dict(pixel_values=pixel_values, input_ids=input_ids, labels=None, attention_mask=attn_mask, position_ids=position_ids,
                            supported_tasks=['captioning' if images is not None and len(images) > 0 else 'textgen'])

def build_vigia_prompt(task, history, tokenizer):
    first_turn = "You are a helpful AI conversational assistant. You can retrieve images or video moments by generating the [RET] token.\nRight now you are helping a user through the task below. Guiden them step by step and answer any questions they may have.\n\n" + task
    convs = [{'from': 'assistant', 'value': first_turn}] + history
    
    img_token = IMG_START_TOKEN + IMG_TOKEN + IMG_END_TOKEN
    for msg in convs:
        msg['value'] = msg['value'].replace('<image>', img_token)
        
    input_text = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)

    return input_text

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
    model, tokenizer, feature_extractor = load_model_and_tokenizer(args)

    dialogues = []
    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    for dialogue in data.keys():
        if any(['ARTIFICIAL.VisualMomentRetrievalIntent' == turn['intent'] for turn in data[dialogue]['dialog']]):
            continue
        dialogues.append(data[dialogue])
        # if len(dialogues) >= args.max_dialogues:
        #     break

    generated_dialogues = []
    with torch.inference_mode():
        stop_tokens = [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.get_vocab().get("<|im_end|>")]

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
                
                prompt = build_vigia_prompt(task, history, tokenizer)
                response = generate_response(model, tokenizer, feature_extractor, prompt, images, stop_tokens)

                history.append({'from': 'assistant', 'value': response})
            generated_dialogues.append({'image': images, 'conversations': history, 'task': dialogue['task']})

    out_path = os.path.join(args.ckpt_path, "test_dialogues_generated.json")
    with open(out_path, "w") as f:
        json.dump(generated_dialogues, f, indent=4)

    print(f"Generated dialogues saved to {out_path}")
