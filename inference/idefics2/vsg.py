import json
import os

import torch
from tqdm import tqdm
from .. import inference_utils
from ..dataset import MMPlanLLMDataset
from time import time
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import copy

FILE_PATH = "/home/dmgcsilva/project/DATA/mmplanllm/conversations_test_vsg.json"

def get_step_text_from_input(data):

    step_text = []
    for item in data:
        input_text = item.get('input', "")

        recipe_steps = [line.split(':')[1] for line in input_text.split('\n') if line.startswith('Step')]

        step_idx = item['target'].lower().split('step ')[-1].strip().split(' ')[0].replace(',', '').replace('.', '').replace(':', '')
        try:
            step_idx = int(step_idx) - 1
            step_text.append(recipe_steps[step_idx])
        except ValueError:
            print(f"Warning: Unable to convert step index '{step_idx}' to int. Skipping.")
            print(f"Target: {item['target']}")
            step_text.append(item['target'])
        except IndexError:
            print(f"Warning: Step index '{step_idx}' out of range for input: {input_text}. Skipping.")
            step_text.append(item['target'])
    return step_text

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceM4/idefics2-8b", help="Model name or path")
    parser.add_argument("--max-samples", type=int, default=1000000, help="Maximum number of samples to process")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")


    return parser.parse_args()


def inference():
    
    args = get_args()

    print(f"Loading model and tokenizer from {args.model}")

    output_file = 'inference/results/{name}_vsg.json'
    name = "idefics2"
    output_file = output_file.format(name=name)


    print(f"Output file: {output_file}")

    # check if parent folder of output file exists
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"Folder {os.path.dirname(output_file)} does not exist. Aborting")
        return

    # load model and tokenizer
    with torch.inference_mode():
        start_time = time()

         # Load the model in half-precision
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = Idefics2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(args.model)

        print(f"Model loaded in {time() - start_time:.2f}s")

        # model in BF16
        model = model.to(dtype=torch.bfloat16)
        # model in eval mode
        model.eval()

        start_time = time()
        # load test dataset
        test_dataset = MMPlanLLMDataset(
            tokenizer=processor, 
            data_path=FILE_PATH,
            max_len=processor.tokenizer.model_max_length,
            is_idefics2=True,
        )

        print(f"Test datasets loaded in {time() - start_time:.2f}s")

        # captioning value holders
        predictions = []

        start_time = time()

        max_samples = min(args.max_samples, len(test_dataset))
        # inference
        with torch.no_grad():
            for idx in tqdm(range(max_samples), desc="Inference"):
                if idx >= max_samples:
                    break

                inputs = test_dataset[idx].to(model.device, torch.float16)

                generate_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.0)

                # get the caption
                gen_step = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
                predictions.append(gen_step)

        print(f"Inference of {max_samples} samples completed in {time() - start_time:.2f}s")
        start_time = time()

        # save predictions
        output_dict = {}
        clean_predictions = [p.replace(s, "").strip() for p, s in zip(predictions, test_dataset.raw_sources)]
        clean_targets = test_dataset.raw_targets[:len(clean_predictions)]
        output_dict['ckpt_path'] = args.model
        output_dict['count'] = max_samples
        output_dict['test_file'] = FILE_PATH
        pred_dicts = []
        for i in range(max_samples):
            pred_dicts.append({
                'input': test_dataset.raw_sources[i],
                'target': test_dataset.raw_targets[i],
                'prediction': clean_predictions[i],
            })

        output_dict['predictions'] = pred_dicts

        # save the targets
        print(f"Saving predictions to {output_file}")
        with open(output_file, "w") as outfile:
            json.dump(output_dict, outfile)

        # calculate metrics
        metrics = dict()
        if os.path.exists(output_file.replace('_vsg.json', '_metrics.json')):
            with open(output_file.replace('_vsg.json', '_metrics.json'), 'r') as f:
                metrics = json.load(f)
        metrics["ckpt"] = "args.model"
        clean_targets = get_step_text_from_input(copy.deepcopy(pred_dicts))

        if not isinstance(clean_targets[0], list):
            clean_targets = [[r] for r in clean_targets]
        if not isinstance(clean_predictions[0], list):
            clean_predictions = [[o] for o in clean_predictions]

        print("Evaluating vsg outputs with rouge...")
        rouge = inference_utils.compute_rouge(clean_targets, clean_predictions)
        print("Evaluating vsg outputs with bleu...")
        bleu = inference_utils.compute_bleu(clean_targets, clean_predictions)
        print("Evaluating vsg outputs with accuracy...")
        accuracy = inference_utils.compute_accuracy(clean_targets, clean_predictions)
        print("Evaluating vsg outputs with exact match...")
        exact_match = inference_utils.calculate_exact_match(clean_targets, clean_predictions)
        print("Evaluating vsg outputs with bertscore...")
        bertscore = inference_utils.compute_bertscore(clean_targets, clean_predictions)

        cap_scores = {
            "rouge": rouge,
            "bleu": bleu,
            "accuracy": accuracy,
            "exact_match": exact_match,
            "bertscore": bertscore
        }

        metrics['vsg'] = cap_scores


        print(metrics)
        with open(output_file.replace('_vsg.json', '_metrics.json'), 'w') as f:
            json.dump(metrics, f)

        print(f"Metrics calculated in {time() - start_time:.2f}s")

if __name__ == "__main__":
    inference()

