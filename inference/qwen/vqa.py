import json
import os

import torch
from tqdm import tqdm
from .. import inference_utils
from ..dataset import MMPlanLLMDataset
from time import time
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

FILE_PATH = "/home/dmgcsilva/project/DATA/mmplanllm/conversations_test_vqa.json"


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
    parser.add_argument("--max-samples", type=int, default=1000000, help="Maximum number of samples to process")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")


    return parser.parse_args()


def inference():
    
    args = get_args()

    output_file = 'inference/results/{name}_vqa.json'
    name = "qwen25"
    output_file = output_file.format(name=name)
    
    print(f"Loading model and tokenizer from {args.model}")


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
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
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
            is_llava=True,
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

                generate_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

                # get the caption
                gen_step = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
                predictions.append(gen_step)

        print(f"Inference of {max_samples} samples completed in {time() - start_time:.2f}s")
        start_time = time()

        # save predictions
        output_dict = {}
        clean_predictions = [p.replace(s, "").strip() for p, s in zip(predictions, test_dataset.raw_sources)]
        clean_targets = test_dataset.raw_targets[:len(clean_predictions)]
        output_dict['ckpt_path'] = "llava-hf/llava-1.5-7b-hf"
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
        # for retrieval, we need to calculate the recall@k with k=1, 5, 10
        # for captioning, we need to calculate the BLEU score and ROUGE score
        metrics = dict()
        if os.path.exists(output_file.replace('_vqa.json', '_metrics.json')):
            with open(output_file.replace('_vqa.json', '_metrics.json'), 'r') as f:
                metrics = json.load(f)
        metrics["ckpt"] = "llava-hf/llava-1.5-7b-hf"
        # retrieval metrics
        

        if not isinstance(clean_targets[0], list):
            clean_targets = [[r] for r in clean_targets]
        if not isinstance(clean_predictions[0], list):
            clean_predictions = [[o] for o in clean_predictions]

        print("Evaluating vqa outputs with rouge...")
        rouge = inference_utils.compute_rouge(clean_targets, clean_predictions)
        print("Evaluating vqa outputs with bleu...")
        bleu = inference_utils.compute_bleu(clean_targets, clean_predictions)
        print("Evaluating vqa outputs with accuracy...")
        accuracy = inference_utils.compute_accuracy(clean_targets, clean_predictions)
        print("Evaluating vqa outputs with exact match...")
        exact_match = inference_utils.calculate_exact_match(clean_targets, clean_predictions)
        print("Evaluating vqa outputs with bertscore...")
        bertscore = inference_utils.compute_bertscore(clean_targets, clean_predictions)

        cap_scores = {
            "rouge": rouge,
            "bleu": bleu,
            "accuracy": accuracy,
            "exact_match": exact_match,
            "bertscore": bertscore
        }

        metrics['vqa'] = cap_scores


        print(metrics)
        with open(output_file.replace('_vqa.json', '_metrics.json'), 'w') as f:
            json.dump(metrics, f)

        print(f"Metrics calculated in {time() - start_time:.2f}s")

if __name__ == "__main__":
    inference()

