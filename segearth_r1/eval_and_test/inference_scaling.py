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
from typing import List, Optional, Callable
from segearth_r1.constants import IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX
from segearth_r1.mm_utils import tokenizer_image_token
from segearth_r1 import conversation as conversation_lib
from segearth_r1.eval_and_test.eval_dataset.RS_val_dataset import preprocess_llama2, preprocess_referring_instruction, DataCollector

PREFIX_INST = "This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:"

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
) -> List[dict]:
    """
    Stage 1 (reason): sample `n_samples` free-text candidate answers for the
    same (image, question) pair at temperature > 0, exactly like the
    "independent parallel generations" step in the paper
    (Sec. 2, "parallel scaling method").
    """
    conv = conversation_lib.conv_templates[conv_version].copy()
    human_turn = f"{PREFIX_INST} {question}"
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

    output_ids = model.generate(
        input_ids_batched,
        attention_mask=attention_mask_batched,
        pad_token_id=pad_token_id,
        images=image_tensor_batched,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    candidates = []
    for i in range(n_samples):
        new_tokens = output_ids[i, input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        candidates.append({"text": text, "n_tokens": int(new_tokens.shape[0])})
    return candidates


def _build_reasoning_item(image, question, answer_text, tokenizer):
    """Mirrors ReasonSegDataset.__getitem__, but with a caller-supplied
    `answer_text` (generated) instead of the ground-truth QA answer."""
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        conversation_lib.default_conversation.version, conversation_lib.default_conversation
    )
    instruction = f" {question}"
    sources = [[
        {"from": "human", "value": PREFIX_INST + "\n<refer>"},
        {"from": "gpt", "value": "Sure, It is <seg>. \n<answer>."},
    ]]
    text_dict = preprocess_llama2(sources, tokenizer)
    input_ids = text_dict["input_ids"][0]
    labels = text_dict["labels"][0]

    token_refer_id = preprocess_referring_instruction(instruction, tokenizer)
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
    probability). Falls back to mean foreground probability if the
    classifier head isn't active (e.g. seg_task == 'referring')."""
    scores = result.get("scores")
    if scores is not None and len(scores) > 0:
        return float(scores.flatten()[0])
    pred = result["pred_masks"]
    pred = pred[0] if pred.dim() == 3 else pred
    fg = pred > 0
    if fg.sum() == 0:
        return 0.0
    return float(pred.sigmoid()[fg].mean())


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
    )

    results = []
    for text, out in zip(answer_texts, outputs):
        pred = out["pred_masks"]
        pred = pred[0] if pred.dim() == 3 else pred
        results.append({
            "text": text,
            "mask": pred.detach().float().cpu(),   # binarized {0,1} HxW
            "score": _mask_confidence(out),
        })
    return results


# --------------------------------------------------------------------------
# Aggregators (mirrors the paper's "average / majority vote / best-of-n /
# worst-of-n" operators, Sec. 2)
# --------------------------------------------------------------------------
def _stack(candidates: List[dict]) -> torch.Tensor:
    return torch.stack([c["mask"] for c in candidates])


def aggregate_average(candidates: List[dict], threshold: float = 0.5) -> torch.Tensor:
    avg = _stack(candidates).mean(0)
    return (avg > threshold).float(), {"mean_conf": float(avg.mean())}


def aggregate_majority_vote(candidates: List[dict]) -> torch.Tensor:
    binary = (_stack(candidates) > 0).float()
    vote = binary.mean(0)
    return (vote > 0.5).float(), {"agreement": float(vote.max())}


def aggregate_best_of_n(
    candidates: List[dict], oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None
) -> torch.Tensor:
    if oracle_iou_fn:
        scored = [(oracle_iou_fn(c["mask"]), c) for c in candidates]
    else:
        scored = [(c["score"], c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return (best["mask"] > 0).float(), {"best_score": scored[0][0]}


def aggregate_worst_of_n(
    candidates: List[dict], oracle_iou_fn: Optional[Callable[[torch.Tensor], float]] = None
) -> torch.Tensor:
    if oracle_iou_fn:
        scored = [(oracle_iou_fn(c["mask"]), c) for c in candidates]
    else:
        scored = [(c["score"], c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    worst = scored[0][1]
    return (worst["mask"] > 0).float(), {"worst_score": scored[0][0]}


_AGGREGATORS = {
    "average": aggregate_average,
    "majority_vote": aggregate_majority_vote,
    "best_of_n": aggregate_best_of_n,
    "worst_of_n": aggregate_worst_of_n,
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
) -> dict:
    best_mask = None
    best_score = -1.0

    for r in range(max_rounds):
        critique = None if r == 0 else "Please reconsider your previous answer and provide a better one."
        answers = sample_reasoning_answers(
            model, tokenizer, image_tensor, question,
            n_samples=1, temperature=temperature, critique_prompt=critique,
        )
        cands = predict_masks_given_answers(
            model, tokenizer, image_tensor, question, [answers[0]["text"]], device=device
        )
        c = cands[0]

        score = oracle_iou_fn(c["mask"]) if oracle_iou_fn else c["score"]
        if score > best_score:
            best_score = score
            best_mask = c["mask"]

    return {
        "mask": (best_mask > 0).float(),
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
    **kwargs
) -> dict:
    if inputs is None:
        inputs = kwargs
    candidates = []
    n_calls = 0
    for i in range(n):
        flipped = random.random() < 0.5
        img = image_tensor.clone()
        if flipped:
            img = img.flip(-1)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_refer_id = inputs["token_refer_id"]
        refer_embedding_indices = inputs["refer_embedding_indices"]
        labels = inputs["labels"]
        out = model.eval_seg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=img.float(),
            masks=None,
            token_refer_id=token_refer_id,
            refer_embedding_indices=refer_embedding_indices,
            labels=labels,
            token_answer_id=None,
            answer_embedding_indices=None,
        )[0]
        n_calls += 1
        pred = out["pred_masks"]
        pred = pred[0] if pred.dim() == 3 else pred
        pred = _flip_mask_back(pred, flipped)
        candidates.append({"mask": pred.detach().float().cpu(), "score": _mask_confidence(out)})
    agg_fn = _AGGREGATORS[aggregator]
    final_mask, agg_info = agg_fn(candidates, oracle_iou_fn=oracle_iou_fn)
    return {
        "mask": final_mask,
        "n_calls": n_calls,
        "candidates": candidates,
        "aggregator": aggregator,
        "aggregator_info": agg_info,
    }


# --------------------------------------------------------------------------
# Convenience: IoU oracle for benchmarking (paper-style "perfect verifier")
# --------------------------------------------------------------------------
def make_iou_oracle(gt_mask: torch.Tensor) -> Callable[[torch.Tensor], float]:
    """Builds an `oracle_iou_fn(pred_mask) -> IoU` closure against a known
    ground-truth mask, for reproducing the paper's best-of-n / worst-of-n
    upper- and lower-bound analysis (Sec. 2, Fig. 3/8) during evaluation.
    Do NOT use this at real deployment time - there is no ground truth then."""
    gt = (gt_mask > 0).float()

    def _iou(pred_mask: torch.Tensor) -> float:
        pred = (pred_mask > 0).float()
        inter = (pred * gt).sum()
        union = (pred + gt).clamp(0, 1).sum()
        return float(inter / (union + 1e-6))

    return _iou
