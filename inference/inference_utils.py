import warnings
import torch

from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_metrics(outputs, references, metrics=["bleu", "meteor", "rouge"]):
    from jury import Jury
    jury_metrics = metrics.copy()
    # replace bleu with local path
    if "bleu" in metrics:
        jury_metrics.remove("bleu")
        jury_metrics.append("./metrics/bleu.py")

    if "accuracy" in metrics:
        jury_metrics.remove("accuracy")
    if "bertscore" in metrics:
        jury_metrics.remove("bertscore")

    scorer = Jury(metrics=jury_metrics)

    outputs = ["gfgfgfgfg" if o == "" else o for o in outputs]
    scores = scorer(predictions=[[o] for o in outputs], references=[[s] for s in references])

    if "accuracy" in metrics:
        scores["accuracy"] = compute_accuracy(outputs, references)

    if "bertscore" in metrics:
        scores['bertscore'] = compute_bertscore(outputs, references)

    return scores

def compute_bleu(outputs, references):
    import sacrebleu
    if isinstance(outputs[0], list):
        outputs = [o[0] for o in outputs]
    if isinstance(references[0], list) and len(references) > 1:
        references = [r[0] for r in references]
    if len(references) > 1:
        references = [references]
    bleu = sacrebleu.metrics.BLEU()
    result = bleu.corpus_score(outputs, references)  # Use corpus_score even for a single sentence
    # get json serializable result
    result = {
        "bleu": result.score,
        "precisions": result.precisions,
        "bp": result.bp,
        "sys_len": result.sys_len,
        "ref_len": result.ref_len,
    }
    return result

def compute_rouge(outputs, references):
    from rouge_score import rouge_scorer
    if isinstance(references[0], list):
        references = [r[0] for r in references]
    if isinstance(outputs[0], list):
        outputs = [o[0] for o in outputs]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores_dicts = [scorer.score(r, o) for r, o in zip(references, outputs)]

    scores = {k: [d[k].fmeasure for d in scores_dicts] for k in scores_dicts[0].keys()}
    scores = {k: sum(v) / len(v) for k, v in scores.items()}

    scores = {k: v * 100 for k, v in scores.items()}

    return scores

def calculate_exact_match(references, outputs):
    if isinstance(references[0], list):
        references = [r[0] for r in references]
    if isinstance(outputs[0], list):
        outputs = [o[0] for o in outputs]

    # partial exact match where if reference is contained in output, it is considered a match
    exact_matches = [1.0 if r.strip().lower() in o.strip().lower() else 0.0 for o, r in zip(outputs, references)]
    exact_match = sum(exact_matches) / len(exact_matches)
    exact_match *= 100
    exact_match = round(exact_match, 2)
    return exact_match

def compute_accuracy(outputs, references):
    if isinstance(references[0], list):
        references = [r[0] for r in references]
    if isinstance(outputs[0], list):
        outputs = [o[0] for o in outputs]
    acc = [1.0 if r.strip().lower() in o.strip().lower() else 0.0 for o, r in zip(outputs, references)]
    acc_mean = sum(acc) / len(acc)
    return acc_mean

def compute_bertscore(outputs, references):
    from bert_score import score
    if isinstance(references[0], list):
        references = [r[0] for r in references]
    if isinstance(outputs[0], list):
        outputs = [o[0] for o in outputs]

    P, R, F1 = score(outputs, references, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=True)

    mean_p = sum(P.tolist()) / len(P.tolist())
    mean_r = sum(R.tolist()) / len(R.tolist())
    mean_f1 = sum(F1.tolist()) / len(F1.tolist())

    return {"recall": mean_r, "precision": mean_p, "f1": mean_f1, "model_type": "microsoft/deberta-xlarge-mnli"}


def get_similarity_matrix(visual_embeds, ret_embeds, type="default"):
    """
    Calculate the similarity matrix between the visual and retrieval embeddings.

    Args:
        visual_embeds: torch.tensor, the visual embeddings with shape (N, D) where N is the number of samples and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (K, D).

    Returns:
        similarity_matrix: torch.tensor, the similarity matrix between the visual and retrieval embeddings with shape (K, N).
    """
    if isinstance(visual_embeds, list):
        visual_embeds = torch.stack(visual_embeds, dim=0)
    if isinstance(visual_embeds, np.ndarray):
        visual_embeds = torch.tensor(visual_embeds)
    if isinstance(ret_embeds, list):
        ret_embeds = torch.stack(ret_embeds, dim=0)
    if isinstance(ret_embeds, np.ndarray):
        ret_embeds = torch.tensor(ret_embeds)

    print("visual_embeds shape: ", visual_embeds.shape)
    print("ret_embeds shape: ", ret_embeds.shape)

    if len(ret_embeds.shape) == 1:
        ret_embeds = ret_embeds.unsqueeze(0)

    if type == "cosine":
        ret_embeds = ret_embeds.unsqueeze(1).cpu()
        visual_embeds = visual_embeds.unsqueeze(0).cpu()

        similarity_matrix = torch.nn.functional.cosine_similarity(ret_embeds, visual_embeds, dim=2)
    else:
        visual_embeds_normalized = torch.nn.functional.normalize(visual_embeds, p=2, dim=1)
        ret_embeds_normalized = torch.nn.functional.normalize(ret_embeds, p=2, dim=1)

        similarity_matrix = torch.matmul(ret_embeds_normalized, visual_embeds_normalized.T).cpu()

    return similarity_matrix


def calculate_iou(target_start, target_end, pred_start, pred_end):
    """
    Calculate the intersection over union (IoU) between two video moments.

    Args:
        target_start: int, the index of the target start frame.
        target_end: int, the index of the target end frame.
        pred_start: int, the index of the predicted start frame.
        pred_end: int, the index of the predicted end frame.

    Returns:
        iou: float, the intersection over union between the two video moments.
    """
    # check if the predicted video moment is valid
    # print("In calculate_iou")
    # print("target_start: ", target_start)
    # print("target_end: ", target_end)
    # print("pred_start: ", pred_start)
    # print("pred_end: ", pred_end)

    if pred_start >= pred_end:
        return 0.0
    if pred_end < target_start or pred_start > target_end:
        return 0.0

    # Calculate the intersection
    intersection_start = max(target_start, pred_start)
    intersection_end = min(target_end, pred_end)

    # Calculate the union
    union_start = min(target_start, pred_start)
    union_end = max(target_end, pred_end)

    # Calculate the overlap
    overlap = max(0, intersection_end - intersection_start)

    # Calculate the IoU
    iou = overlap / (union_end - union_start)

    return iou


def calculate_recall_at_k_custom(visual_embeds, ret_embeds, target_indices, k=[10]):
    similarities = get_similarity_matrix(visual_embeds, ret_embeds)

    recall = dict()

    for k_val in k:
        # Initialize array to hold whether the relevant item is in the top k
        top_k_hits = np.zeros(similarities.shape[0])

        # Iterate over each retrieval embedding
        for i, target_index in enumerate(target_indices):
            # Get indices of top k similar items for this retrieval embedding
            top_k_indices = torch.argsort(similarities[i], descending=True)[:k_val]

            # Check if the target index is in the top k indices
            if target_index in top_k_indices:
                top_k_hits[i] = 1

        y_true = np.ones_like(target_indices)
        recall[f'recall@{k_val}'] = recall_score(y_true, top_k_hits, zero_division=0)

    # scale the recall to 0-100
    recall = {k: v * 100 for k, v in recall.items()}

    return recall

def calculate_R_at_k_IoU_at_m_dynamic_window(
    preds, ids, frame_embs, similarity_threshold, k, m
):
    """
    Calculates the Recall@k IoU=m metric using dynamic window expansion.

    For each query, it finds the top k most similar frames. For each of these
    top k frames, it dynamically expands a window (clip) forwards and backwards
    as long as the frame similarity remains above `similarity_threshold`.
    It checks if any of these k predicted dynamic moments have an IoU >= m
    with the ground truth moment.

    Args:
        preds (list): A list of embeddings [ret_emb, ...].
                      Each emb is a numpy array or list representing the
                      predicted query embedding.
        ids (list): A list of tuples [(start_id, end_id), ...].
                    Each id is an integer ground truth frame index.
        frame_embs (list): A list of lists [[frame_emb_0, frame_emb_1, ...], ...].
                           Each inner list contains the frame embeddings (numpy arrays
                           or lists) for a single video, in order. The outer list
                           corresponds element-wise to `preds` and `ids`.
        similarity_threshold (float): The minimum cosine similarity required for a
                                     frame to be included in the dynamic window
                                     expansion. Should be between -1.0 and 1.0.
        k (int): The number of top predictions (center points) to consider (Recall@k).
        m (float): The IoU threshold (usually between 0.0 and 1.0).

    Returns:
        float: The R@k IoU=m score (average success rate across all queries, scaled to 100).
               Returns 0.0 if `preds` is empty.
    """
    if not preds:
        print("Warning: Input 'preds' list is empty.")
        return 0.0
    if len(preds) != len(ids) or len(preds) != len(frame_embs):
        raise ValueError("Input lists 'preds', 'ids', and 'frame_embs' must have the same length.")

    total_queries = len(preds)
    success_count = 0

    for i, (ret_emb, (gt_start, gt_end), video_frames) in enumerate(zip(preds, ids, frame_embs)):
        # Basic logging, can be removed/adjusted
        # print(f"Processing query {i+1}/{total_queries}...")

        # Ensure embeddings are numpy arrays and have correct shape for cosine_similarity
        try:
            ret_emb_np = np.array(ret_emb).reshape(1, -1)
            video_frames_np = np.array(video_frames)
            if video_frames_np.ndim != 2:
                 raise ValueError(f"Item {i}: video_frames could not be converted to a 2D numpy array (shape: {video_frames_np.shape}).")
            if video_frames_np.shape[0] == 0:
                 warnings.warn(f"Item {i}: Video has no frame embeddings after numpy conversion. Skipping.", UserWarning)
                 continue

        except Exception as e:
            warnings.warn(f"Item {i}: Error processing embeddings - {e}. Skipping.", UserWarning)
            continue

        num_frames = video_frames_np.shape[0]

        # Calculate cosine similarities between predicted emb and all frame embs
        # cosine_similarity returns shape (1, num_frames), so we take [0]
        similarities = cosine_similarity(ret_emb_np, video_frames_np)[0]

        # Get the indices of the top k most similar frames (potential centers)
        # Use argsort in descending order; ensure k isn't larger than num_frames
        actual_k = min(k, num_frames)
        if actual_k <= 0:
             warnings.warn(f"Item {i}: Actual k ({actual_k}) is not positive (num_frames={num_frames}). Skipping this query.", UserWarning)
             continue

        # Indices sorted by similarity (highest first)
        top_k_center_indices = np.argsort(similarities)[::-1][:actual_k]

        found_success_for_this_query = False
        # Iterate through the top k potential center points
        for rank in range(actual_k):
            center_idx = top_k_center_indices[rank]

            # --- Dynamic Window Expansion ---
            pred_start = center_idx
            pred_end = center_idx

            # Expand left (backwards)
            current_start = center_idx
            while True:
                next_start = current_start - 1
                if next_start < 0 or similarities[next_start] < similarity_threshold:
                    break # Stop expanding left: out of bounds or below threshold
                current_start = next_start # Update start index, continue expansion
            pred_start = current_start

            # Expand right (forwards)
            current_end = center_idx
            while True:
                next_end = current_end + 1
                if next_end >= num_frames or similarities[next_end] < similarity_threshold:
                    break # Stop expanding right: out of bounds or below threshold
                current_end = next_end # Update end index, continue expansion
            pred_end = current_end
            # --- End Dynamic Window Expansion ---


            # --- Validity Check ---
            # This check might be less critical now, but good practice.
            # It primarily catches cases where the center_idx itself might be below threshold
            # (unlikely if it's a top-k index unless all similarities are low)
            # or if pred_end calculation somehow stops before pred_start finishes.
            # Also, ensure the clip has some duration. Check if boundaries are inclusive/exclusive.
            # Assuming EXCLUSIVE end boundary for length calculation (consistent with Python slicing)
            # If IoU expects INCLUSIVE, adjust length calculation here and in IoU function.
            if pred_start >= pred_end:
                # This can happen if the window doesn't expand at all AND start == end.
                # A single frame clip might be valid depending on definition, but often IoU needs length > 0.
                # Add a small adjustment if single-frame clips are allowed and IoU handles them.
                # For now, assume clips need at least 1 frame duration (end > start).
                 warnings.warn(
                     f"Item {i}, Rank {rank+1}/{actual_k} (Center Idx: {center_idx}): "
                     f"Predicted start frame ({pred_start}) is not before predicted end frame ({pred_end}) "
                     f"after dynamic expansion. Skipping this candidate.",
                     UserWarning
                 )
                 continue # This specific prediction candidate is invalid or zero-length


            # --- Calculate IoU ---
            # Ensure gt_start, gt_end are also valid
            if gt_start >= gt_end:
                 warnings.warn(
                     f"Item {i}: Ground truth start ({gt_start}) is not before ground truth end ({gt_end}). "
                     f"Cannot calculate valid IoU. Skipping IoU check for this candidate.",
                     UserWarning
                 )
                 iou = 0.0 # Cannot calculate valid IoU
            else:
                 # Using dynamically determined pred_start, pred_end
                 iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)

            # --- Check Condition ---
            if iou >= m:
                found_success_for_this_query = True
                break # Found a successful prediction within the top k, move to next query

        # Increment success count if any of the top k predictions met the criteria
        if found_success_for_this_query:
            success_count += 1

    # Calculate the final average score
    final_score = success_count / float(total_queries) if total_queries > 0 else 0.0
    # Scale the final score to 100
    final_score = final_score * 100
    # Round to 2 decimal places
    final_score = round(final_score, 2)
    return final_score

def calculate_R_at_k_IoU_at_m_with_sim(preds, ids, frame_embs, sim, k, m, return_sim_matrix=False):
    """
    Calculates the Recall@k IoU=m metric for video moment retrieval.

    For each query, it finds the top k predicted moments by pairing the
    top-k most similar start frames and top-k most similar end frames.
    It checks if any of these k predicted moments have an IoU >= m with
    the ground truth moment.

    Args:
        preds (list): A list of tuples [(start_emb, end_emb), ...].
                      Each emb is a numpy array or list representing the
                      predicted start/end frame embedding.
        ids (list): A list of tuples [(start_id, end_id), ...].
                    Each id is an integer ground truth frame index.
        frame_embs (list): A list of lists [[frame_emb_0, frame_emb_1, ...], ...].
                           Each inner list contains the frame embeddings (numpy arrays
                           or lists) for a single video, in order. The outer list
                           corresponds element-wise to `preds` and `ids`.
        sim (int): a similarity threshold to find the most suitable begin and end position
        k (int): The number of top predictions to consider (Recall@k).
        m (float): The IoU threshold (usually between 0.0 and 1.0).
        return_sim_matrix (bool): If True, returns the similarity matrix along with the metrics.

    Returns:
        float: The R@k IoU=m score (average success rate across all queries).
               Returns 0.0 if `preds` is empty.
    """
    if not preds:
        print("Warning: Input 'preds' list is empty.")
        return 0.0
    if len(preds) != len(ids) or len(preds) != len(frame_embs):
        raise ValueError("Input lists 'preds', 'ids', and 'frame_embs' must have the same length.")

    total_queries = len(preds)
    success_count = 0
    start_sim_matrixes = [] # for each sample, store a similarity array of the representation against all frames
    end_sim_matrixes = []
    predictions = []

    for i, ((start_emb, end_emb), (gt_start, gt_end), video_frames) in enumerate(zip(preds, ids, frame_embs)):
        # print(f"Processing query {i+1}/{total_queries}...")
        # print(f"Inputs: {start_emb.shape}, {end_emb.shape}, {gt_start}, {gt_end}, {video_frames.shape}")
        # Ensure embeddings are numpy arrays and have correct shape for cosine_similarity
        try:
            start_emb_np = np.array(start_emb).reshape(1, -1)
            end_emb_np = np.array(end_emb).reshape(1, -1)
            video_frames_np = np.array(video_frames)
            if video_frames_np.ndim == 1:
                 # If frames are 1D, assume it's a list of scalars (unlikely) or needs reshape
                 # This depends heavily on the actual embedding structure.
                 # Assuming embeddings are vectors, video_frames_np should be 2D.
                 raise ValueError(f"Item {i}: video_frames could not be converted to a 2D numpy array.")
            if video_frames_np.shape[0] == 0:
                 warnings.warn(f"Item {i}: Video has no frame embeddings after numpy conversion. Skipping.", UserWarning)
                 continue

        except Exception as e:
            # warnings.warn(f"Item {i}: Error processing embeddings - {e}. Skipping.", UserWarning)
            # continue
            raise e

        num_frames = video_frames_np.shape[0]

        # Calculate cosine similarities between predicted embs and all frame embs
        # cosine_similarity returns shape (1, num_frames), so we take [0]
        start_similarities = cosine_similarity(start_emb_np, video_frames_np)[0]
        end_similarities = cosine_similarity(end_emb_np, video_frames_np)[0]
        # start_similarities = get_similarity_matrix(video_frames_np, start_emb_np, type="default")[0]
        # end_similarities = get_similarity_matrix(video_frames_np, end_emb_np, type="default")[0]

        # Store the similarity matrices for later use
        start_sim_matrixes.append(start_similarities)
        end_sim_matrixes.append(end_similarities)

        # Get the indices of the top k most similar frames for start and end
        # Use argsort in descending order; ensure k isn't larger than num_frames
        actual_k = min(k, num_frames)
        # if actual_k <= 0: # Should only happen if num_frames is 0, handled above, but good practice
        #      warnings.warn(f"Item {i}: Actual k ({actual_k}) is not positive. Skipping this query.", UserWarning)
        #      continue

        # Indices sorted by similarity (highest first)
        top_k_start_indices = np.argsort(start_similarities)[::-1][:actual_k]
        top_k_end_indices = np.argsort(end_similarities)[::-1][:actual_k]
        # print(f"Top k start indices: {top_k_start_indices}")
        # print(f"Top k end indices: {top_k_end_indices}")

        found_success_for_this_query = False
        # Iterate through the top k paired predictions
        for rank in range(actual_k):
            pred_start = top_k_start_indices[rank]
            pred_end = top_k_end_indices[rank]

            # --- Dynamic Window Expansion ---
            # Expand left (backwards)
            current_start = pred_start
            while True:
                next_start = current_start - 1
                if next_start < 0 or start_similarities[next_start] < sim:
                    break # Stop expanding left: out of bounds or below threshold
                current_start = next_start # Update start index, continue expansion
            pred_start = current_start

            # Expand right (forwards)
            current_end = pred_end
            while True:
                next_end = current_end + 1
                if next_end >= num_frames or end_similarities[next_end] < sim:
                    break # Stop expanding right: out of bounds or below threshold
                current_end = next_end # Update end index, continue expansion
            pred_end = current_end

            if actual_k == 1:
                predictions.append((pred_start, pred_end))

            # --- Validity Check ---
            if pred_start >= pred_end:
                # warnings.warn(
                #     f"DYNAMIC: Item {i}, Rank {rank+1}/{actual_k}: Predicted start frame ({pred_start}) "
                #     f"is not before predicted end frame ({pred_end}). Skipping this pair.",
                #     UserWarning
                # )
                continue # This specific prediction pair is invalid

            # --- Calculate IoU ---
            iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)

            # --- Check Condition ---
            if iou >= m:
                found_success_for_this_query = True
                break # Found a successful prediction within the top k, move to next query

        # Increment success count if any of the top k predictions met the criteria
        if found_success_for_this_query:
            success_count += 1

    # Calculate the final average score
    final_score = success_count / float(total_queries) if total_queries > 0 else 0.0
    # scale the final score to 100
    final_score = final_score * 100
    # round to 2 decimal places
    final_score = round(final_score, 2)

    if return_sim_matrix:
        # Return the similarity matrices along with the final score
        return start_sim_matrixes, end_sim_matrixes, predictions, final_score
    return final_score

def calculate_R_at_k_IoU_at_m(preds, ids, frame_embs, k, m):
    """
    Calculates the Recall@k IoU=m metric for video moment retrieval.

    For each query, it finds the top k predicted moments by pairing the
    top-k most similar start frames and top-k most similar end frames.
    It checks if any of these k predicted moments have an IoU >= m with
    the ground truth moment.

    Args:
        preds (list): A list of tuples [(start_emb, end_emb), ...].
                      Each emb is a numpy array or list representing the
                      predicted start/end frame embedding.
        ids (list): A list of tuples [(start_id, end_id), ...].
                    Each id is an integer ground truth frame index.
        frame_embs (list): A list of lists [[frame_emb_0, frame_emb_1, ...], ...].
                           Each inner list contains the frame embeddings (numpy arrays
                           or lists) for a single video, in order. The outer list
                           corresponds element-wise to `preds` and `ids`.
        k (int): The number of top predictions to consider (Recall@k).
        m (float): The IoU threshold (usually between 0.0 and 1.0).

    Returns:
        float: The R@k IoU=m score (average success rate across all queries).
               Returns 0.0 if `preds` is empty.
    """
    if not preds:
        print("Warning: Input 'preds' list is empty.")
        return 0.0
    if len(preds) != len(ids) or len(preds) != len(frame_embs):
        raise ValueError("Input lists 'preds', 'ids', and 'frame_embs' must have the same length.")

    total_queries = len(preds)
    success_count = 0

    for i, ((start_emb, end_emb), (gt_start, gt_end), video_frames) in enumerate(zip(preds, ids, frame_embs)):
        print(f"Processing query {i+1}/{total_queries}...")
        print(f"Inputs: {start_emb.shape}, {end_emb.shape}, {gt_start}, {gt_end}, {video_frames.shape}")
        # Ensure embeddings are numpy arrays and have correct shape for cosine_similarity
        try:
            start_emb_np = np.array(start_emb).reshape(1, -1)
            end_emb_np = np.array(end_emb).reshape(1, -1)
            video_frames_np = np.array(video_frames)
            if video_frames_np.ndim == 1:
                 # If frames are 1D, assume it's a list of scalars (unlikely) or needs reshape
                 # This depends heavily on the actual embedding structure.
                 # Assuming embeddings are vectors, video_frames_np should be 2D.
                 raise ValueError(f"Item {i}: video_frames could not be converted to a 2D numpy array.")
            if video_frames_np.shape[0] == 0:
                 warnings.warn(f"Item {i}: Video has no frame embeddings after numpy conversion. Skipping.", UserWarning)
                 continue

        except Exception as e:
            # warnings.warn(f"Item {i}: Error processing embeddings - {e}. Skipping.", UserWarning)
            # continue
            raise e

        num_frames = video_frames_np.shape[0]

        # Calculate cosine similarities between predicted embs and all frame embs
        # cosine_similarity returns shape (1, num_frames), so we take [0]
        start_similarities = cosine_similarity(start_emb_np, video_frames_np)[0]
        end_similarities = cosine_similarity(end_emb_np, video_frames_np)[0]
        # start_similarities = get_similarity_matrix(video_frames_np, start_emb_np, type="default")[0]
        # end_similarities = get_similarity_matrix(video_frames_np, end_emb_np, type="default")[0]

        # Get the indices of the top k most similar frames for start and end
        # Use argsort in descending order; ensure k isn't larger than num_frames
        actual_k = min(k, num_frames)
        if actual_k <= 0: # Should only happen if num_frames is 0, handled above, but good practice
             warnings.warn(f"Item {i}: Actual k ({actual_k}) is not positive. Skipping this query.", UserWarning)
             continue

        # Indices sorted by similarity (highest first)
        top_k_start_indices = np.argsort(start_similarities)[::-1][:actual_k]
        top_k_end_indices = np.argsort(end_similarities)[::-1][:actual_k]
        # print(f"Top k start indices: {top_k_start_indices}")
        # print(f"Top k end indices: {top_k_end_indices}")

        found_success_for_this_query = False
        # Iterate through the top k paired predictions
        for rank in range(actual_k):
            pred_start = top_k_start_indices[rank]
            pred_end = top_k_end_indices[rank]

            # --- Validity Check ---
            if pred_start >= pred_end:
                warnings.warn(
                    f"Item {i}, Rank {rank+1}/{actual_k}: Predicted start frame ({pred_start}) "
                    f"is not before predicted end frame ({pred_end}). Skipping this pair.",
                    UserWarning
                )
                continue # This specific prediction pair is invalid

            # --- Calculate IoU ---
            iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)

            # --- Check Condition ---
            if iou >= m:
                found_success_for_this_query = True
                break # Found a successful prediction within the top k, move to next query

        # Increment success count if any of the top k predictions met the criteria
        if found_success_for_this_query:
            success_count += 1

    # Calculate the final average score
    final_score = success_count / float(total_queries) if total_queries > 0 else 0.0
    # scale the final score to 100
    final_score = final_score * 100
    # round to 2 decimal places
    final_score = round(final_score, 2)
    return final_score


def calculate_recall_at_k_vmr(target_visual_embeds, ret_embeds, target_frames, k=[10]):
    """
    Calculate the recall@k for the video moment retrieval task.
    This function supports multiple target frames

    Args:
        target_visual_embeds: list(torch.tensor), the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[list[Dict]], a list of lists where each inner list contains the indices of all acceptable frames
        k: list, the k values for which to calculate the recall@k.

    Returns:
        recall_at_k: list, the recall@k values for each k.
    """
    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    recall = dict()

    for i in range(len(target_frames)):
        assert len(target_frames[i]) > 0, f"Target frames must contain at least one frame for sample {i}"
        r = calculate_recall_at_k_vmr_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i], k)

        for k_val in k:
            if f'recall@{k_val}' not in recall:
                recall[f'recall@{k_val}'] = []
            recall[f'recall@{k_val}'].append(r[f'recall@{k_val}'])

    # Calculate the mean recall@k
    for k_val in k:
        recall[f'recall@{k_val}'] = sum(recall[f'recall@{k_val}']) / len(recall[f'recall@{k_val}'])
        recall[f'recall@{k_val}'] = recall[f'recall@{k_val}'] * 100

    return recall


def calculate_recall_at_k_vmr_individual(target_visual_embeds, ret_embeds, target_indices, k=[10]):
    """
    Calculate the recall@k for the video moment retrieval task for a single sample.

    Args:
        target_visual_embeds: torch.tensor, the visual embeddings for the video frames with shape (M, D) where M is the number of frames in the video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (D).
        target_indices: list[Dict], a list of dictionaries where each dictionary contains the indices of the target start and end frames.
        k: list, the k values for which to calculate the recall@k.

    Returns:
        recall_at_k: list, the recall@k values for each k.
    """

    # Calculate the similarities between retrieval and visual embeddings
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)
    recall = dict()

    for k_val in k:
        # Get the top k indices
        top_k = torch.argsort(similarities, descending=True)[:k_val]

        # Calculate the recall@k
        correct = 0
        for target_index in target_indices:
            if target_index in top_k:
                correct += 1

        # Compute the recall@k knowing that the target is a single video moment
        recall[f"recall@{k_val}"] = correct / min(k_val, len(target_indices))

    return recall


def calculate_recall_at_k_w_iou_version2(target_visual_embeds, ret_embeds, target_frames, iou=0.7, k=[10]):
    """
        Calculate the recall@k for the video moment retrieval task with intersection over union (IoU) constraint.
        Here we first calculate the retrieved video moment middle frame embeddings and then calculate the IoU between
        the retrieved video moment and the target video moment. The retrieved video moment is calculated by assuming a
        clip of the same size as the gt centered on the retrieved frame. If the IoU is greater than the specified
        threshold, we consider the retrieval successful.

        Args:
            target_visual_embeds: torch.tensor, the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
            ret_embeds: torch.tensor, the retrieval embeddings for the start frames with shape (N, D).
            target_frames: list[Dict], a list of dictionaries where each dictionary contains the indices of the target start and end frames.
            iou: float, the intersection over union threshold.
            k: list, the k values for which to calculate the recall@k.

        Returns:
            recall_at_k: list, the recall@k values for each k.
        """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    recall = dict()

    for i in range(len(target_frames)):
        assert target_frames[i]["start"] <= target_frames[i]["end"], f"Start frame index must be less than end frame index for target frame {i} but got {target_frames[i]['start']} and {target_frames[i]['end']}"

        r = calculate_recall_at_k_w_iou_version2_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i], iou, k)

        for k_val in k:
            if f'recall@{k_val}_wiou@{iou}' not in recall:
                recall[f'recall@{k_val}_wiou@{iou}'] = []
            recall[f'recall@{k_val}_wiou@{iou}'].append(r[f'recall@{k_val}_wiou@{iou}'])

    # Calculate the mean recall@k
    for k_val in k:
        recall[f'recall@{k_val}_wiou@{iou}'] = sum(recall[f'recall@{k_val}_wiou@{iou}']) / len(recall[f'recall@{k_val}_wiou@{iou}'])

    return recall

def calculate_recall_at_k_w_iou_version2_individual(target_visual_embeds, ret_embeds, target_indices, iou, k=[10]):
    """
        Calculate the recall@k for the video moment retrieval task with intersection over union (IoU) constraint for a single sample.
        Args:
            target_visual_embeds: torch.tensor, the visual embeddings for the video frames with shape (M, D) where M is the number of frames in the video, and D is the dimensionality of the embeddings.
            ret_embeds: torch.tensor, the retrieval embeddings for the end frames with shape (, D).
            target_frames: Dict, a dictionary containing the indices of the target start and end frames.
            iou: float, the intersection over union threshold.
            k: list, the k values for which to calculate the recall@k.

        Returns:
            recall_at_k: list, the recall@k values for each k.
        """

    # Calculate the similarities between retrieval and visual embeddings
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)
    recall = dict()

    for k_val in k:
        # Get the top k indices
        top_k = torch.argsort(similarities, descending=True)[:k_val]

        # Calculate the IoU for each pair of start and end frames
        ious = []
        for i in range(k_val):
            middle_frame = top_k[i]
            gt_size = max(1, target_indices["end"] - target_indices["start"])
            start_frame = max(0, middle_frame - (gt_size // 2))
            end_frame = min(target_visual_embeds.shape[0] - 1, middle_frame + (gt_size // 2))
            iou_val = calculate_iou(target_indices["start"], target_indices["end"], start_frame, end_frame)
            ious.append(iou_val)

        # Calculate the recall@k
        correct = sum([_iou >= iou for _iou in ious])
        # Compute the recall@k knowing that the target is a single video moment
        recall[f"recall@{k_val}_wiou@{iou}"] = 1 if correct > 0 else 0

    return recall

def calculate_avg_overlap(target_visual_embeds, ret_embeds, target_frames):
    """
    Calculate the average overlap between the target and retrieved video moments.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[Dict], a list of dictionaries where each dictionary contains the indices of the target start and end frames.

    Returns:
        avg_overlap: float, the average overlap between the target and retrieved video moments.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    overlaps = []

    for i in range(len(target_frames)):
        assert target_frames[i]["start"] <= target_frames[i]["end"], f"Start frame index must be less than end frame index for target frame {i} but got {target_frames[i]['start']} and {target_frames[i]['end']}"
        overlaps.append(calculate_overlap_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i]))

    avg_overlap = sum(overlaps) / len(overlaps)

    avg_overlap = float(avg_overlap)

    return avg_overlap

def calculate_overlap_individual(target_visual_embeds, ret_embeds, target_indices):
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)

    top_frame = torch.argsort(similarities, descending=True)[0]

    gt_size = target_indices["end"] - target_indices["start"] - 1
    start_frame = max(0, top_frame - (gt_size // 2))
    end_frame = min(target_visual_embeds.shape[0] - 1, top_frame + (gt_size // 2))

    overlap = calculate_iou(target_indices["start"], target_indices["end"], start_frame, end_frame)

    return overlap

def calculate_map_vmr(target_visual_embeds, ret_embeds, target_frames):
    """
    Calculate the mean average precision for the video moment retrieval task.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[int], a list of integers where each integer is the index of the target frame.

    Returns:
        map: float, the mean average precision for the video moment retrieval task.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    aps = []

    for i in range(len(target_frames)):
        assert isinstance(target_frames[i], int), f"Target frames must contain a single frame index for sample {i}"
        aps.append(calculate_map(target_visual_embeds[i], ret_embeds[i].unsqueeze(0), [target_frames[i]]))

    map = sum(aps) / len(aps)

    # map = map * 100

    return map


def calculate_frame_distance(target_visual_embeds, ret_embeds, target_frames):
    """
    Get the average frame distance between the target frame and the retrieved frame.
    To make this resistant to outliers, we will calculate the median of the frame distances.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[int], a list of integers where each integer is the index of the target frame.

    Returns:
        avg_frame_distance: float, the average frame distance between the target frame and the retrieved frame.
        median_frame_distance: float, the median frame distance between the target frame and the retrieved frame.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    frame_distances = []
    video_sizes = [embed.shape[0] for embed in target_visual_embeds]

    for i in range(len(target_frames)):
        assert isinstance(target_frames[i], int), f"Target frames must contain a single frame index for sample {i}"
        frame_distances.append(calculate_avg_frame_distance_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i]))

    # Normalizing frame distances by their respective video sizes
    normd_frame_distances = np.array(frame_distances) / np.array(video_sizes)

    scores = {
        "avg_frame_distance": np.mean(frame_distances),
        "normd_avg_frame_distance": np.mean(normd_frame_distances),
        "median_frame_distance": np.median(frame_distances)
    }

    return scores

def calculate_avg_frame_distance_individual(target_visual_embeds, ret_embeds, target_index):
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)

    top_frame = torch.argsort(similarities, descending=True)[0]

    return abs(top_frame - target_index)

def calculate_step_accuracy(target_visual_embeds, ret_embeds, target_frames):
    """
    Get the video moment accuracy between the target frame and the retrieved frame, meaning if the top retrieved frame is within the target frame.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[Dict], a list of dictionaries that contain the start and end frame of the relevant video moment.

    Returns:
        step_accuracy: float, the step accuracy between the target frame and the retrieved frame.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    step_accuracies = []

    for i in range(len(target_frames)):
        assert len(target_frames[i]) > 0, f"Target frames must contain at least one frame for sample {i}"
        step_accuracies.append(calculate_step_accuracy_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i]))

    step_accuracy = sum(step_accuracies) / len(step_accuracies)

    step_accuracy = step_accuracy * 100

    return step_accuracy

def calculate_step_accuracy_individual(target_visual_embeds, ret_embeds, target_frames):
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)

    top_frame = torch.argsort(similarities, descending=True)[0]

    if target_frames["start"] <= top_frame <= target_frames["end"]:
        return 1
    return 0



def save_similarity_matrix(visual_embeds, ret_embeds, save_path):
    """
    Calculate the similarity matrix between the visual and retrieval embeddings and save it to disk.

    Args:
        visual_embeds: torch.tensor, the visual embeddings with shape (N, D) where N is the number of samples and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        save_path: str, the path where the similarity matrix will be saved.
    """
    # Calculate similarities between retrieval and visual embeddings
    # Assuming cosine similarity for calculation, normalize embeddings first
    similarities = get_similarity_matrix(visual_embeds, ret_embeds)

    # Save the similarity matrix to disk
    torch.save(similarities, save_path)

    print(f"Similarity matrix saved to {save_path}")


def load_similarity_matrix(similarity_matrix_path):
    """
    Load a similarity matrix from disk.

    Args:
        similarity_matrix_path: str, the path to the saved similarity matrix.

    Returns:
        similarity_matrix: torch.tensor, the loaded similarity matrix.
    """
    return torch.load(similarity_matrix_path)


def calculate_map(visual_embeds, ret_embeds, target_indices):
    """
    Calculate the mean average precision for the retrieval task.
    This function assumes a single relevant image per query.

    Args:
        all_vis_embeds: torch.tensor, the visual embeddings with shape (N, D) where N is the number of samples and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_indices: list, the indices of the target images in the visual_embeds. (N)

    Returns:
        map: float, the mean average precision.
    """
    similarities = get_similarity_matrix(visual_embeds, ret_embeds)

    # Calculate the average precision for each query
    aps = []
    for i, target_index in enumerate(target_indices):
        # Sort the similarities
        sorted_similarities = torch.argsort(similarities[i], descending=True)
        # Calculate the precision
        aps.append(1 / (torch.where(sorted_similarities == target_index)[0].item() + 1))
    # Calculate the mean average precision
    map = sum(aps) / len(aps)

    # scale the map to 0-100
    map = map * 100

    return map

def get_topk_retrievals(target_visual_embeds, ret_embeds, k=10):
    """
    Get the top k indices retrieved, for each sample.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        k: int, the number of top retrievals to get.

    Returns:
        top_k_indices: list, the top k indices for each sample.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]} and {target_visual_embeds.shape[0]}"

    top_k_indices = []

    for i in range(ret_embeds.shape[0]):
        similarities = get_similarity_matrix(target_visual_embeds[i], ret_embeds[i]).squeeze(0)

        top_frames = torch.argsort(similarities, descending=True)
        top_frames = top_frames[:min(k, len(top_frames))].cpu().numpy()

        sims = {
            "indices": top_frames,
            "sim_scores": similarities[top_frames]
        }

        top_k_indices.append(sims)

    return top_k_indices

