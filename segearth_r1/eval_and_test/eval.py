import os
import cv2
import torch
from enum import Enum
from tqdm import tqdm
import numpy as np
from eval_dataset.RS_val_dataset import DataCollector, RRSISDDataset, ReasonSegDataset, RefSegRSDataset
from segearth_r1.eval_and_test.eval_dataset.liss4_val_dataset import Liss4ReasonSegDataset
from segearth_r1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX ,ANSWER_TOKEN_INDEX
from segearth_r1.model.builder import load_pretrained_model
from segearth_r1.utils import disable_torch_init
from segearth_r1.mm_utils import get_model_name_from_path
from segearth_r1 import conversation as conversation_lib
from torch.utils.data import DataLoader
from typing import Optional
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)
    
def compute_metric(intersection_meter, union_meter, acc_iou_meter, pr_meters, cur_res, gt):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i, result in enumerate(cur_res):
        gt_mask = gt[i].squeeze(0).int().cuda().contiguous()
        pred_masks = result["pred_masks"].int().cuda().contiguous()

        scores = result["scores"]
        if scores is not None and len(scores) > 0:
            # scores is already a Tensor from SEG_instance_inference — no need to re-wrap
            scores = scores.detach().float() if isinstance(scores, torch.Tensor) else torch.tensor(scores, dtype=torch.float)
            top_idx = torch.topk(scores, 1).indices.cpu().numpy()
            preds_to_eval = pred_masks[top_idx, :]
        else:
            preds_to_eval = [pred_masks]

        max_acc_iou = -1
        best_iou = None
        best_intersection = best_union = None

        for pred in preds_to_eval:
            intersection, union, _ = intersectionAndUnionGPU(pred, gt_mask, 2, ignore_index=255)
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()

            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  
            foreground_iou = acc_iou[1]

            if foreground_iou > max_acc_iou:
                max_acc_iou = foreground_iou
                best_iou = acc_iou
                best_intersection = intersection
                best_union = union

        intersection_meter.update(best_intersection)
        union_meter.update(best_union)
        acc_iou_meter.update(best_iou, n=1)

        for threshold in thresholds:
            pr_meters[threshold].update(1.0 if best_iou[1] > threshold else 0.0, n=1)
        
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    base_data_path: str = "/root/siton-data-412581749c3f4cfea0d7c972b8742057/data"
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./segearth_r1/mask_config/maskformer2_swin_large.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r1'
    version: str = 'llava_phi'
    segmentation: bool = True
    eval_batch_size: int = 5
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")
    data_split: Optional[str] = field(default="val")
    use_seg_query: bool = False
    dataset_type: Optional[str] = field(default="RRSIS-D")
    vis_path: Optional[str] = field(default=None)
    scaling_type: Optional[str] = field(default=None) #For parallel_reasoning, sequential_reasoning or parallel_referring
    scaling_n: int = field(default=8) #For number of samples (N) for parallel_reasoning: Number of text-and-mask samples generated; sequential_reasoning: The maximum number of self-critique rounds
    # parallel_referring: Determines how many image resolutions/scales to process in TTA.
    scaling_temp: float = field(default=1.0) #Amount of variation in the generated samples.
    scaling_aggregator: str = field(default="average")       # 'average', 'majority_vote', or 'best_of_n'
    flip_prob: float = field(default=0.5, metadata={"help": "Probability of horizontal flip for test-time augmentation (TTA) in referring segmentation."})
    use_oracle: bool = field(default=False, metadata={"help": "If True, pass ground-truth IoU oracle to best_of_n/worst_of_n for upper-bound analysis. Never use for real benchmark numbers."})

def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = data_args.model_path
    model_name = get_model_name_from_path(model_path)
    print(f'current model is {model_path}')
    tokenizer, model, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, 
                                                          use_seg_query = data_args.use_seg_query, device='cuda')
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    if data_args.dataset_type == 'RRSIS-D':
        eval_dataset = RRSISDDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split = data_args.data_split,
        )
    elif data_args.dataset_type == 'EarthReason':
        eval_dataset = ReasonSegDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split=data_args.data_split,
        )
    elif data_args.dataset_type == 'Liss4Reason':
        eval_dataset = Liss4ReasonSegDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split=data_args.data_split,
        )
    elif data_args.dataset_type == 'RefSegRS':
        eval_dataset = RefSegRSDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split=data_args.data_split,
        )
    else:
        raise ValueError(f"Unknown dataset_type: '{data_args.dataset_type}'. "
                         f"Expected one of: RRSIS-D, EarthReason, Liss4Reason, RefSegRS")
    data_collator = DataCollector(
        tokenizer=tokenizer,
    )
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use same dtype as training (bf16 if supported, else fp16)
    eval_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(device=device, dtype=eval_dtype).eval()
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    pr_meters = {
        threshold: AverageMeter(f"Pr@{threshold}", ":6.3f", Summary.AVERAGE)
        for threshold in thresholds
    }
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            gt = inputs["masks"]
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            if getattr(data_args, 'scaling_type', None) is not None:
                from segearth_r1.eval_and_test.inference_scaling import (
                    parallel_scale_reasoning,
                    sequential_scale_reasoning,
                    parallel_scale_referring,
                    make_iou_oracle
                )
                outputs = []
                # Process each item in the batch sequentially for ensembling
                for b_idx in range(inputs['images'].shape[0]):
                    image_tensor = inputs['images'][b_idx:b_idx+1]
                    gt_mask_item = gt[b_idx].to(device)
                    # Only use the oracle IoU function for explicit upper-bound analysis.
                    # Passing oracle_iou_fn when use_oracle=False would let best_of_n/worst_of_n
                    # cheat by selecting candidates using ground-truth labels.
                    oracle_iou_fn = make_iou_oracle(gt_mask_item) if data_args.use_oracle else None
                    
                    if data_args.scaling_type == "parallel_reasoning":
                        # Decode the pre-tokenized question; drop the trailing [SEG] marker
                        # that preprocess_referring_instruction always appends.
                        question_ids = inputs['token_refer_id'][b_idx]
                        question = tokenizer.decode(question_ids[:-1], skip_special_tokens=True).strip()
                        
                        scaled_res = parallel_scale_reasoning(
                            model, tokenizer, image_tensor, question,
                            n=data_args.scaling_n, temperature=data_args.scaling_temp,
                            aggregator=data_args.scaling_aggregator, device=device,
                            oracle_iou_fn=oracle_iou_fn
                        )
                        # Format output to match the structure compute_metric expects
                        outputs.append({"pred_masks": scaled_res["mask"].cuda(), "scores": None})
                        
                    elif data_args.scaling_type == "sequential_reasoning":
                        question_ids = inputs['token_refer_id'][b_idx]
                        question = tokenizer.decode(question_ids[:-1], skip_special_tokens=True).strip()
                        
                        scaled_res = sequential_scale_reasoning(
                            model, tokenizer, image_tensor, question,
                            max_rounds=data_args.scaling_n, temperature=data_args.scaling_temp,
                            device=device, oracle_iou_fn=oracle_iou_fn
                        )
                        outputs.append({"pred_masks": scaled_res["mask"].cuda(), "scores": None})
                        
                    elif data_args.scaling_type == "parallel_referring":
                        question_ids = inputs['token_refer_id'][b_idx]
                        referring_text = tokenizer.decode(question_ids[:-1], skip_special_tokens=True).strip()
                        
                        scaled_res = parallel_scale_referring(
                            model, tokenizer, image_tensor,
                            input_ids=inputs['input_ids'][b_idx:b_idx+1],
                            attention_mask=inputs['attention_mask'][b_idx:b_idx+1],
                            token_refer_id=[inputs['token_refer_id'][b_idx]],
                            refer_embedding_indices=inputs['refer_embedding_indices'][b_idx:b_idx+1],
                            labels=inputs['labels'][b_idx:b_idx+1],
                            n=data_args.scaling_n,
                            aggregator=data_args.scaling_aggregator,
                            flip_prob=data_args.flip_prob,
                            referring_text=referring_text,
                            device=device,
                            oracle_iou_fn=oracle_iou_fn
                        )
                        outputs.append({"pred_masks": scaled_res["mask"].cuda(), "scores": None})
            else:
                if 'token_answer_id' in inputs:
                    inputs['token_answer_id'] = [ids.to(device) for ids in inputs['token_answer_id']]
                    outputs = model.eval_seg(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        images=inputs['images'].float(),
                        masks=inputs['masks'],
                        token_refer_id = inputs['token_refer_id'],
                        refer_embedding_indices=inputs['refer_embedding_indices'],
                        labels=inputs['labels'],
                        token_answer_id=inputs['token_answer_id'],
                        answer_embedding_indices=inputs['answer_embedding_indices']
                        )
                else:
                    outputs = model.eval_seg(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        images=inputs['images'].float(),
                        masks=inputs['masks'],
                        token_refer_id = inputs['token_refer_id'],
                        refer_embedding_indices=inputs['refer_embedding_indices'],
                        labels=inputs['labels'],
                        token_answer_id=None,
                        answer_embedding_indices=None
                        )
            # vis
            if data_args.vis_path is not None:
                os.makedirs(data_args.vis_path, exist_ok=True)
                for vis_idx, image_name in enumerate(inputs['image_name']):
                    gt_mask = inputs['masks'][vis_idx].squeeze(0) * 255
                    pred_mask = outputs[vis_idx]['pred_masks'] * 255
                    if pred_mask.dim() == 3:
                        pred_mask = pred_mask[0]   # take top-scoring query for multi-mask outputs
                    gt_mask_np = gt_mask.cpu().numpy().astype(np.uint8)
                    pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
                    gt_mask_root = os.path.join(data_args.vis_path, image_name + "_gt.png")
                    pred_mask_root = os.path.join(data_args.vis_path, image_name + "_pred.png")
                    cv2.imwrite(gt_mask_root, gt_mask_np)
                    cv2.imwrite(pred_mask_root, pred_mask_np)

            compute_metric(intersection_meter,union_meter,acc_iou_meter, pr_meters, outputs, gt)
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] 
    giou = acc_iou_meter.avg[1] 
    print(
            "ciou: {:.4f}, giou: {:.4f}".format(ciou, giou)
        )
    print(
            "IoU Thresholds: " + 
            ", ".join([f"@{t}: {m.avg:.4f}" for t, m in pr_meters.items()])
        )
    
if __name__ == "__main__":
    evaluation()
