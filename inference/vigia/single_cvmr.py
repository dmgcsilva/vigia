import json
import os
import numpy as np
import warnings

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ..model import IMG_TOKEN, RET_TOKEN, RET_TOKEN_2, VIGIA
from ..dataset import MMPlanLLMDataset
from ..inference_utils import calculate_R_at_k_IoU_at_m, calculate_R_at_k_IoU_at_m_dynamic_window, calculate_R_at_k_IoU_at_m_with_sim
from time import time

FILE_PATH = "/home/dmgcsilva/project/DATA/mmplanllm/conversations_test_cvmr.json"


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
        output_file = f"./{output_file}_cvmr.json"
    else:
        output_file = os.path.join(args.ckpt_path, f'{output_file}_cvmr.json')
    
    if 'diy' in FILE_PATH:
        output_file = output_file.replace('cvmr', 'diy_cvmr')
    if 'recipe' in FILE_PATH:
        output_file = output_file.replace('cvmr', 'recipe_cvmr')

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
        ret_embs_preds = []
        target_embs = []
        target_ids = []
        start_time = time()

        max_samples = min(args.max_samples, len(test_dataset))
        # inference
        with torch.no_grad():
            for idx in tqdm(range(max_samples), desc="Inference"):
                if idx >= max_samples:
                    break

                item = test_dataset[idx]
                input_ids = item['input_ids'].to(device=model.model.lm.device)
                text_embs = model.model.input_embeddings(item['input_ids'].to(device=model.model.lm.device).unsqueeze(0))
                assert input_ids[-1] == model.model.end_ret_token_id, f"End RET token not found in input_ids: {input_ids}"
                assert input_ids[-2] == model.model.start_ret_token_id, f"Start RET token not found in input_ids: {input_ids}"
                _, output_embeddings, _ = model( # TODO: Modify generate function to deal with dual retrieval output
                    text_embs,
                    None,
                    None,
                    generate=True,
                    num_words=1,  # give it a maximum of 5 words to generate RET
                    temperature=0.0,
                )

                # get the last token embeddings for the RET token after passing through the start and end projection heads
                start_ret_emb, end_ret_emb = output_embeddings[0]  # shape (1, seq_len, hidden_size)
                start_ret_emb = start_ret_emb[:, -2, :]  # shape (1, hidden_size)
                end_ret_emb = end_ret_emb[:, -1, :]  # shape (1, hidden_size)

                ret_embs_preds.append((start_ret_emb.cpu().detach().float().numpy(), end_ret_emb.cpu().detach().float().numpy()))
                
                ids, tgt_embs = test_dataset.get_target_embeddings(idx, model)
                target_embs.append(tgt_embs.cpu().detach().float().numpy())
                target_ids.append(ids)

        print(f"Inference of {max_samples} samples completed in {time() - start_time:.2f}s")
        start_time = time()

        print("target_ids", target_ids)
        print("target_embs", target_embs[0].shape) # (N, 512)
        print("ret_embs_preds", ret_embs_preds[0][0].shape) # (1, 512)
        # calculate metrics
        metrics = dict()
        if os.path.exists(output_file.replace('_cvmr.json', '_metrics.json')):
            with open(output_file.replace('_cvmr.json', '_metrics.json'), 'r') as f:
                metrics = json.load(f)
        metrics["ckpt"] = args.ckpt_path
        
        # retrieval metrics
        # all_visual_embeds = torch.tensor(target_embs).to(device).squeeze(1).squeeze(1).squeeze(1)
        # all_ret_embeds = torch.tensor(ret_embs_preds).to(device).squeeze(1)

        # drop every key in metrics that has cvmr
        for k in list(metrics.keys()):
            if "cvmr" in k:
                del metrics[k]

        metrics_inner = {}
        for k in [1, 5, 10]:
            for m in [0.5, 0.7, 1.0]:
                r_at_k_iou_at_m = calculate_R_at_k_IoU_at_m(ret_embs_preds, target_ids, target_embs, k=k, m=m)
                metrics_inner[f"R@{k}_IoU@{m}"] = r_at_k_iou_at_m


        metrics['cvmr'] = metrics_inner

        # find the boundaries dynamically
        for sim in [0.35, 0.3, 0.25]:
            metrics_inner = {}
            for k in [1, 5, 10]:
                for m in [0.5, 0.7, 1.0]:
                    r_at_k_iou_at_m_w_sim = calculate_R_at_k_IoU_at_m_with_sim(ret_embs_preds, target_ids, target_embs, sim=sim, k=k, m=m)
                    metrics_inner[f'R@{k}_IoU@{m}'] = r_at_k_iou_at_m_w_sim
            
            metrics[f'cvmr_sim@{sim}'] = metrics_inner

        # assert all_visual_embeds.shape[0] == all_ret_embeds.shape[0]
        tmp_preds = [r[0] for r in ret_embs_preds]
        for t in [0.35, 0.3, 0.25]:
            metrics_inner = {}
            for k in [1, 5, 10]:
                for m in [0.5, 0.7, 1.0]:
                    r_at_k_iou_at_m = calculate_R_at_k_IoU_at_m_dynamic_window(tmp_preds, target_ids, target_embs, t, k=k, m=m)
                    metrics_inner[f"R@{k}_IoU@{m}"] = r_at_k_iou_at_m


            metrics[f'start_cvmr@{t}'] = metrics_inner
        tmp_preds = [r[1] for r in ret_embs_preds]
        for t in [0.35, 0.3, 0.25]:
            metrics_inner = {}
            for k in [1, 5, 10]:
                for m in [0.5, 0.7, 1.0]:
                    r_at_k_iou_at_m = calculate_R_at_k_IoU_at_m_dynamic_window(tmp_preds, target_ids, target_embs, t, k=k, m=m)
                    metrics_inner[f"R@{k}_IoU@{m}"] = r_at_k_iou_at_m


            metrics[f'end_cvmr@{t}'] = metrics_inner

        # use mean
        tmp_preds = [(r[0] + r[1]) / 2 for r in ret_embs_preds]
        # for t in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        for t in [0.35, 0.3, 0.25]:
            metrics_inner = {}
            for k in [1, 5, 10]:
                for m in [0.5, 0.7, 1.0]:
                    r_at_k_iou_at_m = calculate_R_at_k_IoU_at_m_dynamic_window(tmp_preds, target_ids, target_embs, t, k=k, m=m)
                    metrics_inner[f"R@{k}_IoU@{m}"] = r_at_k_iou_at_m


            metrics[f'mean_cvmr@{t}'] = metrics_inner
        print(metrics)
        with open(output_file.replace('_cvmr.json', '_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # save the similarity matrixes for calculate_R_at_k_IoU_at_m_with_sim with m=0.35
        start_sim_matrixes, end_sim_matrixes, predictions, r_at_k_iou_at_m_w_sim = calculate_R_at_k_IoU_at_m_with_sim(ret_embs_preds, target_ids, target_embs, sim=0.35, k=1, m=0.5, return_sim_matrix=True)
        print(f"length of start_sim_matrixes: {len(start_sim_matrixes)}, length of end_sim_matrixes: {len(end_sim_matrixes)}")
        # save pickle file
        sim_matrixes = {
            "paths": [item['image'][0] for item in test_dataset.data[:max_samples]],
            "start_sim_matrixes": start_sim_matrixes,
            "end_sim_matrixes": end_sim_matrixes,
            "predictions": predictions,
            "target_ids": target_ids
        }
        torch.save(sim_matrixes, output_file.replace('_cvmr.json', '_sim_matrixes.pth'))
        print(f"Similarity matrixes saved to {output_file.replace('_cvmr.json', '_sim_matrixes.pth')}")

        print(f"Metrics calculated in {time() - start_time:.2f}s")

if __name__ == "__main__":
    inference()

