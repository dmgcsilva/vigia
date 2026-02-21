import json
import os

import torch
from tqdm import tqdm
from .. import inference_utils
from transformers import AutoTokenizer

from ..model import IMG_TOKEN, RET_TOKEN, RET_TOKEN_2, VIGIA
from ..dataset import MMPlanLLMDataset
from time import time

FILE_PATH = "/home/dmgcsilva/project/DATA/mmplanllm/conversations_test_vqa.json"


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--max-samples", type=int, default=1000000, help="Maximum number of samples to process")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")


    return parser.parse_args()


def inference():
    
    args = get_args()

    print(f"Loading model and tokenizer from {args.ckpt_path}")

    args.ckpt_path = args.ckpt_path[:-1] if args.ckpt_path[-1] == '/' else args.ckpt_path

    output_file = args.ckpt_path.split('/')[-1]

    if not os.path.exists(args.ckpt_path):
        output_file = f"./{output_file}_vqa.json"
    else:
        output_file = os.path.join(args.ckpt_path, f'{output_file}_vqa.json')

    print(f"Output file: {output_file}")

    # check if parent folder of output file exists
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"Folder {os.path.dirname(output_file)} does not exist. Aborting")
        return

    # load model and tokenizer
    with torch.inference_mode():
        start_time = time()

        model = VIGIA.from_pretrained(args.ckpt_path, ignore_mismatched_sizes=True)
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True, trust_remote_code=True)
        assert tokenizer.get_vocab().get(RET_TOKEN) == model.model.start_ret_token_id, f"Retrieval token mismatch: {tokenizer.get_vocab().get(RET_TOKEN)} != {model.model.start_ret_token_id}"
        assert tokenizer.get_vocab().get(RET_TOKEN_2) == model.model.end_ret_token_id, f"Retrieval token mismatch: {tokenizer.get_vocab().get(RET_TOKEN_2)} != {model.model.end_ret_token_id}"
        assert tokenizer.get_vocab().get(IMG_TOKEN) == model.model.img_token_id, f"Image token mismatch: {tokenizer.get_vocab().get(IMG_TOKEN)} != {model.model.img_token_id}"

        print(f"Model loaded in {time() - start_time:.2f}s")


        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)

        # model in BF16
        model = model.to(dtype=torch.bfloat16)
        # model in eval mode
        model.eval()

        tokenizer.model_max_length = 2048
        stop_tokens = [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.get_vocab().get("<|im_end|>")]

        start_time = time()
        # load test dataset
        test_dataset = MMPlanLLMDataset(
            tokenizer=tokenizer, 
            data_path=FILE_PATH,
            feature_extractor_model=model.config.visual_encoder,
            max_len=tokenizer.model_max_length,
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

                item = test_dataset[idx]
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
                predictions.append(gen_step)

        print(f"Inference of {max_samples} samples completed in {time() - start_time:.2f}s")
        start_time = time()

        # save predictions
        output_dict = {}
        clean_predictions = [p.replace(s, "").strip() for p, s in zip(predictions, test_dataset.raw_sources)]
        clean_targets = test_dataset.raw_targets[:len(clean_predictions)]
        output_dict['ckpt_path'] = args.ckpt_path
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
        metrics["ckpt"] = args.ckpt_path
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
            json.dump(metrics, f, indent=4)

        print(f"Metrics calculated in {time() - start_time:.2f}s")

if __name__ == "__main__":
    inference()

