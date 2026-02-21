from vllm import LLM, SamplingParams
from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import re
from PIL import Image
from tqdm import tqdm
import os
import torch

PROMPT="""Evaluate the following dialogue between a user and an assistant.
The assistant is an AI language and vision model designed to guide the users through a given procedural plan (e.g., a recipe or a DIY task).
The assistant should be helpful, informative, and accurate in following the steps of the plan. It should not make up steps or provide incorrect information. It also should be polite but concise, avoiding unnecessary repetition and commentary. 
Please rate the overall quality of the assistant's responses, at the dialogue level, on a scale from 1 to 5, where 1 means very poor and 5 means excellent.
Keep in mind that there could be some jumps in the steps if the user provides an image for context that indicates further progress than covered in the dialogue (eg, the user last asked for step 4 but ended up doing 4, 5 and 6 before asking for help again that is conveyed via a picture of their current progress). Take this into account when rating the assistant's performance (was the jump awarranted based on the image provided? Did the assistant handle the jump well?).

Here is the task the dialogue is based on:
{task_description}

Here is the dialogue:
{dialogue}

Please provide your rating as an integer from 1 to 5, feel free to reason about it before giving the final score.
For automatic parsing, please end your response with "SCORE: X", where X is your rating.
"""

PROMPTV2="""You are an expert evaluator for Multimodal Procedural Guidance Assistants.
Your task is to evaluate a full dialogue session between a User and a VLM Assistant.

For the purpose of this evalution you are provided with:
- TASK: The official ground-truth procedural plan (recipe, manual, etc.).
- Dialogue Transcript: A chronological log of the interaction. Each turn contains:
   - User Input: Text and (optional) Image Description.
   - Assistant Response: The response you must evaluate.
   - Ground-Truth Response: The ground truth response (useful for checking state/facts).

Your task is to evaluate the entire interaction on these 3 dimensions (Score 1-5, where 1 is poor, 3 is acceptable, and 5 is excellent):

1 - State Tracking
   - Does the Assistant correctly identify the user's progress through the TASK based on the User's text and images?
   - If the User silently skips steps (evident in images or ground-truth response), does the Assistant correctly recognize this and jumps to the new step?
   - Penalize drifting from the plan or failing to recognize visual completion of steps, but do so proportionally to the severity of the error.

2 - Succinctness
   - Does the Assistant provide direct, actionable instructions without excessive conversational filler or unwanted commentary?
   - The goal is that the Assistant should be a tool, not a chatty companion. It should only speak when necessary to guide or warn.
   - Do not penalize politeness or empathy.

3 - Plan Adherence
   - Does the Assistant remain faithful to the TASK?
   - Penaltize hallucinating tools, ingredients, or steps that do not exist in the TASK.

OUTPUT FORMAT
1 - First, provide a concise reasoning block analyzing the dialogue. You may critique specific turns.
2 - End with a valid JSON block, with the following structure:
{{
  "state_tracking_score": int,
  "succinctness_score": int,
  "plan_adherence_score": int,
}}

Here is the TASK:
{task_description}

Here is the dialogue:
{dialogue}

Please provide your evaluation below."""

TURN_PROMPT = """Turn {turn_number}:
User: {user_text}
Ground-Truth Response: {gold_response}
Model Response {model_response}"""

def load_model():
    model_name = "/home/dmgcsilva/project/DATA/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a"
    model = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        max_model_len=16384,
        max_num_batched_tokens=16384,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_task_from_dialogue(data):
    if 'recipe' in data:
        recipe = data.get('recipe', {})
    elif 'task' in data:
        recipe = data.get('task', {})
    else:
        raise ValueError(f"No recipe found in task data: {data}") 
    
    title = recipe.get('displayName', 'Untitled Recipe')
    instructions = recipe.get('instructions', [])
    
    output_lines = ["Task: " + title]
    for index, step in enumerate(instructions, 1):
        step_text = step.get('stepText', '').strip()
        output_lines.append(f"Step {index}: {step_text}")
    
    return "\n".join(output_lines)

def evaluate_dialogue(model, dialogue):
    
    assert 'task' in dialogue and 'conversations' in dialogue and 'image' in dialogue, "Dialogue must contain 'task', 'conversations', and 'image' fields."
    task_description = get_task_from_dialogue(dialogue['task'])
    conversations = dialogue['conversations']
    images = dialogue['image']
    dialogue_text = ""
    for turn_idx in range(0, len(conversations), 2):
        dialogue_text += TURN_PROMPT.format(
            turn_number = (turn_idx // 2) + 1,
            user_text = conversations[turn_idx]['value'],
            gold_response = dialogue['reference_answers'][turn_idx // 2],
            model_response = conversations[turn_idx + 1]['value'],
        ) + "\n\n"

    prompt = PROMPTV2.format(task_description=task_description, dialogue=dialogue_text)
    prompt = prompt.replace("<im_start>[IMG]<im_end>", "<start_of_image>\n")
    prompt = prompt.replace("<image>", "<start_of_image>\n")

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
    )

    inputs = {
        "prompt": f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "multi_modal_data": {"image": [Image.open(img_path).convert('RGB') for img_path in images]},
    }

    # print(f"First Input: \n{inputs['prompt']}")
    # print(f"Image count: {len(images)}")
    # print(f"Image tokens in prompt: {inputs['prompt'].count('<start_of_image>')}")

    response = model.generate(inputs, sampling_params=sampling_params)
    # print(f"Response: {response[0].outputs[0].text.strip()}")
    return response[0].outputs[0].text.strip()

def parse_score(response):
    # extract json block from response
    json_block = re.search(r"\{.*\}", response, re.DOTALL)
    if json_block is None:
        return None
    try:
        scores = json.loads(json_block.group(0))
        if all(k in scores for k in ['state_tracking_score', 'succinctness_score', 'plan_adherence_score']):
            return scores
        else:
            return None
    except json.JSONDecodeError:
        return None

if __name__ == "__main__":
    # start by clearing the GPU memory
    torch.cuda.empty_cache()

    parser = ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True, help="Path to the input text file.")
    args = parser.parse_args()

    model, tokenizer = load_model()

    with open(args.file_path, "r") as f:
        dialogues = json.load(f)

    results = []
    for dialogue in tqdm(dialogues):
        response = evaluate_dialogue(model, dialogue)
        score = parse_score(response)
        if score is None:
            response = evaluate_dialogue(model, dialogue)
            score = parse_score(response)
            if score is None:
                score = {"state_tracking_score": 0, "succinctness_score": 0, "plan_adherence_score": 0}
                print(f"Could not parse score for dialogue. Response: {response}")
            
        results.append({
            "dialogue": dialogue,
            "model_response": response,
            "scores": score
        })

    out_path = args.file_path.replace(".json", "_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Average Scores:")
    print(f" - State Tracking: {sum([r['scores']['state_tracking_score'] for r in results]) / len(results):.2f}")
    print(f" - Succinctness: {sum([r['scores']['succinctness_score'] for r in results]) / len(results):.2f}")
    print(f" - Plan Adherence: {sum([r['scores']['plan_adherence_score'] for r in results]) / len(results):.2f}")


"""
PROMPT 1 EVAL RESULTS
VIGIA           3.48
QWEN 2.5 VL     3.31
LLAVA OV        2.39
QWEN 3 VL       4.06
INTERNVL 3.5    3.5

PROMPT V2 EVAL RESULTS (GEMMA 3 27B IT)
VIGIA (full context)
 - State Tracking: 2.94
 - Succinctness: 3.15
 - Plan Adherence: 4.09
QWEN 2.5 VL
 - State Tracking: 1.70
 - Succinctness: 1.74
 - Plan Adherence: 2.61
LLAVA OV
 - State Tracking: 1.17
 - Succinctness: 2.48
 - Plan Adherence: 2.54
QWEN 3 VL
 - State Tracking: 2.98
 - Succinctness: 2.06
 - Plan Adherence: 3.81
INTERNVL 3.5
 - State Tracking: 2.02
 - Succinctness: 1.98
 - Plan Adherence: 3.13
MMPlanLLM (full context)
 - State Tracking: 1.09
 - Succinctness: 3.02
 - Plan Adherence: 2.07
"""
    