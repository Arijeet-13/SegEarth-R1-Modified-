"""
Inference-time scaling for SegEarth-R1
=======================================
Adapts the three strategies from "Inference-Time Scaling for Complex Tasks:
Where We Stand and What Lies Ahead" (Balachandran et al., 2025, MSR,
arXiv:2504.00294) to geospatial pixel reasoning / referring segmentation:

  1. Standard single pass   -> predict_mask_given_answer / predict_mask_referring
  2. Parallel scaling       -> parallel_scale_reasoning / parallel_scale_referring
                                aggregators: average | majority_vote | best_of_n | worst_of_n
  3. Sequential scaling     -> sequential_scale_reasoning
                                (iterative self-critique using the model's own
                                confidence, or ground-truth IoU when benchmarking)
"""

import torch
import torch.nn.functional as F
import random
from typing import List, Optional, Callable, Tuple, Dict
from segearth_r1.constants import IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX
from segearth_r1.mm_utils import tokenizer_image_token
from segearth_r1 import conversation as conversation_lib
from segearth_r1.eval_and_test.eval_dataset.RS_val_dataset import preprocess_llama2, preprocess_referring_instruction, DataCollector

PREFIX_INST = "This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:"

def _binarize(mask: torch.Tensor) -> torch.Tensor:
    """Ensures consistent pixel-level binarization across all functions."""
    if mask.min() >= 0.0 and mask.max() <= 1.0:
        return (mask > 0.5).float()
    return (mask > 0.0).float()

@torch.no_grad()
def sample_reasoning_answers(
    model,
    tokenizer,
    image_tensor: torch.Tensor,       # [1, 3, H, W], already preprocessed
    question: str,
    n_samples: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
    conv_version: str = "llava_phi",
    critique_prompt: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> List[dict]:
    """
    Stage 1 (reason): sample `n_samples` free-text candidate answers for the
    same (image, question) pair at temperature > 0. Supports threading conversation history.
    """
    conv = conversation_lib.conv_templates[conv_version].copy()
    
    # Inject conversation history turns
    if history:
        for role, content in history:
            conv.append_message(role, content)
            
    human_turn = f"{PREFIX_INST} {question}" if not history else question
    if critique_prompt:
        human_turn += f"\n{critique_prompt}"
        
    conv.append_message(conv.roles[0], human_turn)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(image_tensor.device)
    
    # Run all generations in parallel as a single batch pass
    input_ids_batched = input_ids.repeat(n_samples, 1)
    attention_mask_batched = torch.ones_like(input_ids_batched)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    image_tensor_batched = image_tensor.repeat(n_samples, 1, 1, 1)

    chunk = 4
    all_ids = []
    for i in range(0, n_samples, chunk):
        g = min(chunk, n_samples - i)
        out = model.generate(
            input_ids_batched[i:i+g],
            attention_mask=attention_mask_batched[i:i+g],
            pad_token_id=pad_token_id,
            images=image_tensor_batched[i:i+g],
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        all_ids.append(out)
        torch.cuda.empty_cache()
    output_ids = torch.cat(all_ids, dim=0)

    candidates = []
    for i in range(n_samples):
        new_tokens = output_ids[i, input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        candidates.append({"text": text, "n_tokens": int(new_tokens.shape[0])})
    return candidates


def _build_reasoning_item(image, question, answer_text, tokenizer):
    """Mirrors ReasonSegDataset.__getitem__, but with a caller-supplied
    `answer_text` (generated) instead of the ground-truth QA answer."""
    # Ensure default_conversation is set to the phi template used throughout eval.
    # Use a direct key lookup rather than round-tripping through .version, since
    # conv_llava_phi.version == "phi" which is not a key in conv_templates.
    if conversation_lib.default_conversation.version != "phi":
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_phi"]
    instruction = f" {question}"
    sources = [[
        {"from": "human", "value": PREFIX_INST + "\n<refer>"},
        {"from": "gpt", "value": "Sure, It is <seg>. \n<answer>."},
    ]]
    text_dict = preprocess_llama2(sources, tokenizer)
    input_ids = text_dict["input_ids"][0]
    labels = text_dict["labels"][0]

    token_refer_id = preprocess_referring_instruction(instruction, tokenizer)
    # Match training: append [SEG] to answer
    token_answer_id = preprocess_referring_instruction(answer_text, tokenizer)
    refer_embedding_indices = torch.zeros_like(input_ids)
    refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
    answer_embedding_indices = torch.zeros_like(input_ids)
    answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "image": image,
        "image_name": "scaled_infer",
        "dataset_type": "reason_seg",
        "token_refer_id": token_refer_id,
        "token_answer_id": token_answer_id,
        "refer_embedding_indices": refer_embedding_indices,
        "answer_embedding_indices": answer_embedding_indices,
    }


def _mask_confidence(result: dict) -> float:
    """Extracts the free confidence signal SegEarth-R1 already computes in
    `SEG_instance_inference` (SEG-classifier score x mean foreground mask
    probability). Falls back to mean sigmoid probability if the
    classifier head isn't active (e.g. seg_task == 'referring')."""
    scores = result.get("scores")
    if scores is not None and len(scores) > 0:
        # Use the maximum score across all candidate queries, not an arbitrary index
        return float(scores.max())

    # Fall back to mean sigmoid probability over raw logits (not binarized masks)
    raw = result.get("raw_masks")
    if raw is None:
        print("[_mask_confidence] WARNING: No raw_masks available, returning 0.0")
        return 0.0  # no confidence signal available
    pred = raw[0] if raw.dim() == 3 else raw
    conf = float(pred.sigmoid().mean())
    # Debug: print actual logit statistics
    print(f"[_mask_confidence DEBUG] raw logit range: [{pred.min():.3f}, {pred.max():.3f}], mean: {pred.mean():.3f}, confidence: {conf:.6f}")
    return conf  # proper confidence from logits



@torch.no_grad()
def predict_masks_given_answers(
    model,
    tokenizer,
    image_tensor: torch.Tensor,   # [1, 3, H, W] (single image, will be repeated)
    question: str,
    answer_texts: List[str],
    device="cuda",
) -> List[dict]:
    """
    Stage 2 (segment): batches all candidate answer strings for ONE image
    into a single `eval_seg` call and returns one {mask, score} per
    candidate.
    """
    image_cpu = image_tensor.squeeze(0).cpu()
    items = [
        _build_reasoning_item(image_cpu, question, ans, tokenizer)
        for ans in answer_texts
    ]
    batch = DataCollector(tokenizer=tokenizer)(items)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["token_refer_id"] = [t.to(device) for t in batch["token_refer_id"]]
    batch["token_answer_id"] = [t.to(device) for t in batch["token_answer_id"]]

    # Pre-warm the vision-tower cache with the single image so the backbone
    # runs exactly once regardless of how many candidates are in the batch.
    model._cached_raw_features = None
    with torch.no_grad():
        _ = model.run_vision_tower(batch["images"].float())

    outputs = model.eval_seg(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=batch["images"].float(),
        masks=None,
        token_refer_id=batch["token_refer_id"],
        refer_embedding_indices=batch["refer_embedding_indices"],
        labels=batch["labels"],
        token_answer_id=batch["token_answer_id"],
        answer_embedding_indices=batch["answer_embedding_indices"],
        _reuse_vision_cache=True,
    )

    results = []
    for text, out in zip(answer_texts, outputs):
        pred = out["pred_masks"]
        pred = pred[0] if pred.dim() == 3 else pred
        raw = out.get("raw_masks")
        if raw is not None:
            raw = raw[0] if raw.dim() == 3 else raw
            raw = raw.detach().cpu()
        results.append({
            "text": text,
            "mask": _binarize(pred.detach().cpu()),   # consistently binarized {0,1} HxW
            "raw_masks": raw,
            "score": _mask_confidence(out),
        })
    return results


# --------------------------------------------------------------------------
# Aggregators
# --------------------------------------------------------------------------
def _stack(candidates: List[dict]) -> torch.Tensor:
    return torch.stack([c["mask"] for c in candidates])


def aggregate_average(candidates, threshold=0.5, **kwargs):
    if all(c.get("raw_masks") is not None for c in candidates):
        probs = torch.stack([c["raw_masks"].sigmoid() for c in candidates]).mean(0)
    else:
        probs = _stack(candidates).mean(0)
    return (probs > threshold).float(), {"mean_conf": float(probs.mean())}


def aggregate_majority_vote(candidates: List[dict], **kwargs) -> Tuple[torch.Tensor, dict]:
    binary = _stack(candidates)
    vote = binary.mean(0)
    return (vote > 0.5).float(), {"agreement": float(vote.mean())}


def aggregate_best_of_n(
    candidates: List[dict], oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None, **kwargs
) -> Tuple[torch.Tensor, dict]:
    if oracle_iou_fn:
        scored = [(oracle_iou_fn(c["mask"]), c) for c in candidates]
    else:
        scored = [(c["score"], c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] <= 1e-6:
        print("[best_of_n] all candidates scored ~0 — degenerate batch")
    best = scored[0][1]
    return best["mask"], {"best_score": scored[0][0]}


def aggregate_worst_of_n(
    candidates: List[dict], oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None, **kwargs
) -> Tuple[torch.Tensor, dict]:
    if oracle_iou_fn:
        scored = [(oracle_iou_fn(c["mask"]), c) for c in candidates]
    else:
        scored = [(c["score"], c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    worst = scored[0][1]
    return worst["mask"], {"worst_score": scored[0][0]}


def _consensus_score(candidates, idx):
    own = candidates[idx]["mask"]
    others = [c["mask"] for j, c in enumerate(candidates) if j != idx]
    if not others:
        return candidates[idx]["score"]
    consensus = (torch.stack(others).mean(0) > 0.5).float()
    inter = (own * consensus).sum()
    union = own.sum() + consensus.sum() - inter
    return float(inter / (union + 1e-8))

def aggregate_consensus_best_of_n(candidates, oracle_iou_fn=None, **kwargs):
    if oracle_iou_fn:
        return aggregate_best_of_n(candidates, oracle_iou_fn=oracle_iou_fn)
    scored = [(_consensus_score(candidates, i), c) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]["mask"], {"consensus_score": scored[0][0]}

_AGGREGATORS = {
    "average": aggregate_average,
    "majority_vote": aggregate_majority_vote,
    "best_of_n": aggregate_best_of_n,
    "worst_of_n": aggregate_worst_of_n,
    "consensus_best_of_n": aggregate_consensus_best_of_n,
}


# --------------------------------------------------------------------------
# Main Scaling Pipelines
# --------------------------------------------------------------------------
def parallel_scale_reasoning(
    model,
    tokenizer,
    image_tensor: torch.Tensor,
    question: str,
    n: int = 8,
    temperature: float = 1.0,
    aggregator: str = "best_of_n",
    device="cuda",
    oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None,
) -> dict:
    answers = sample_reasoning_answers(
        model, tokenizer, image_tensor, question,
        n_samples=n, temperature=temperature,
    )
    answer_texts = [a["text"] for a in answers]
    candidates = predict_masks_given_answers(
        model, tokenizer, image_tensor, question, answer_texts, device=device
    )

    agg_fn = _AGGREGATORS[aggregator]
    final_mask, agg_info = agg_fn(candidates, oracle_iou_fn=oracle_iou_fn)
    return {
        "mask": final_mask, "n_calls": len(candidates), "candidates": candidates,
        "aggregator": aggregator, "aggregator_info": agg_info,
    }


def sequential_scale_reasoning(
    model,
    tokenizer,
    image_tensor: torch.Tensor,
    question: str,
    max_rounds: int = 4,
    temperature: float = 0.7,
    device="cuda",
    oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None,
    conv_version: str = "llava_phi",
    max_history_turns: int = 4,
) -> dict:
    best_mask = None
    best_score = -1.0

    conv = conversation_lib.conv_templates[conv_version].copy()
    history = []

    for r in range(max_rounds):
        critique = None if r == 0 else "Please reconsider your previous answer and provide a better one."
        # Trim history to avoid exceeding context length: keep last max_history_turns pairs
        trimmed_history = history[-(max_history_turns * 2):] if len(history) > max_history_turns * 2 else history
        answers = sample_reasoning_answers(
            model, tokenizer, image_tensor, question,
            n_samples=1, temperature=temperature, critique_prompt=critique,
            history=trimmed_history, conv_version=conv_version,
        )
        ans_text = answers[0]["text"]
        cands = predict_masks_given_answers(
            model, tokenizer, image_tensor, question, [ans_text], device=device
        )
        c = cands[0]

        score = oracle_iou_fn(c["mask"]) if oracle_iou_fn else c["score"]
        if score > best_score:
            best_score = score
            best_mask = c["mask"]

        # Record the conversation turns into history for the next iteration round
        # Don't include critique in history - it's added fresh each round by sample_reasoning_answers
        user_msg = f"{PREFIX_INST} {question}" if r == 0 else question
        history.append((conv.roles[0], user_msg))
        history.append((conv.roles[1], ans_text))

    return {
        "mask": best_mask,
        "n_calls": max_rounds,
        "best_score": best_score,
    }


def _flip_mask_back(mask: torch.Tensor, flipped: bool) -> torch.Tensor:
    return mask.flip(-1) if flipped else mask


def parallel_scale_referring(
    model,
    tokenizer,
    image_tensor: torch.Tensor,
    inputs: Optional[dict] = None,
    n: int = 8,
    aggregator: str = "average",
    device="cuda",
    oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None,
    flip_prob: float = 0.5,   # default matches DataArguments; 0.0 gives zero diversity
    referring_text: Optional[str] = None,
    **kwargs
) -> dict:
    if inputs is None:
        inputs = kwargs

    # Determine whether we can safely flip the image (avoid directional terms like left/right/east/west)
    directional_words = ["left", "right", "east", "west", "top", "bottom", "north", "south", "upper", "lower"]
    should_flip = flip_prob > 0.0
    if should_flip and referring_text is not None:
        if any(w in referring_text.lower() for w in directional_words):
            should_flip = False

    if not should_flip and n > 1:
        # No augmentation possible — all passes would be identical; run just once and return.
        import warnings
        warnings.warn(
            "[parallel_scale_referring] flip_prob=0 or directional text detected: "
            "all n passes are identical. Running a single pass instead.",
            RuntimeWarning, stacklevel=2,
        )
        n = 1

    # Stack inputs into a batch for a single forward pass
    input_ids = inputs["input_ids"].repeat(n, 1)
    attention_mask = inputs["attention_mask"].repeat(n, 1)
    refer_embedding_indices = inputs["refer_embedding_indices"].repeat(n, 1)
    labels = inputs["labels"].repeat(n, 1) if inputs.get("labels") is not None else None
    token_refer_id = inputs["token_refer_id"] * n

    images_list = []
    flipped_flags = []
    for i in range(n):
        do_flip = should_flip and (random.random() < flip_prob)
        img = image_tensor.clone()
        if do_flip:
            img = img.flip(-1)
        images_list.append(img)
        flipped_flags.append(do_flip)

    images_batched = torch.cat(images_list, dim=0) # [n, 3, H, W]

    # Evaluate the batched candidates in a single pass
    # Vision cache is not reused here since images may differ (flips).
    # eval_seg will run the backbone once on the full batch.
    outputs = model.eval_seg(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images_batched.float(),
        masks=None,
        token_refer_id=token_refer_id,
        refer_embedding_indices=refer_embedding_indices,
        labels=labels,
        token_answer_id=None,
        answer_embedding_indices=None,
    )

    candidates = []
    for out, flipped in zip(outputs, flipped_flags):
        pred = out["pred_masks"]
        pred = pred[0] if pred.dim() == 3 else pred
        pred = _flip_mask_back(pred, flipped)
        raw = out.get("raw_masks")
        if raw is not None:
            raw = raw[0] if raw.dim() == 3 else raw
            raw = _flip_mask_back(raw.detach().cpu(), flipped)
        candidates.append({
            "mask": _binarize(pred.detach().cpu()),
            "raw_masks": raw,
            "score": _mask_confidence(out),
        })

    agg_fn = _AGGREGATORS[aggregator]
    final_mask, agg_info = agg_fn(candidates, oracle_iou_fn=oracle_iou_fn)
    return {
        "mask": final_mask,
        "n_calls": n,
        "candidates": candidates,
        "aggregator": aggregator,
        "aggregator_info": agg_info,
    }


def make_iou_oracle(gt_mask: torch.Tensor) -> Callable[[torch.Tensor], float]:
    """Builds an `oracle_iou_fn(pred_mask) -> IoU` closure against a known
    ground-truth mask, for reproducing the paper's best-of-n / worst-of-n
    upper- and lower-bound analysis."""
    gt = (gt_mask.cpu() > 0).float()

    def _iou(pred_mask: torch.Tensor) -> float:
        pred = (pred_mask > 0).float()
        inter = (pred * gt).sum()
        union = (pred + gt).clamp(0, 1).sum()
        return float(inter / (union + 1e-6))

    return _iou
