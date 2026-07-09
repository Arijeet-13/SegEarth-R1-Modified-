import os
import torch
import shutil
import copy
from transformers import Trainer
from segearth_r1.eval_and_test.inference_scaling import _build_reasoning_item, PREFIX_INST, DataCollector
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.distributed as dist
from typing import Optional
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available, is_torch_tpu_available,is_accelerate_available
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union



import torch

from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class LLaVATrainer(Trainer):
    def compute_iou_reward(self, pred_mask, gt_mask):
        """Task-driven outcome reward (Intersection over Union)."""
        pred = (pred_mask > 0.5).float()
        gt = (gt_mask > 0.5).float()
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        return float(intersection / (union + 1e-8))

    def compute_soft_iou_reward(self, pred_prob, gt_mask):
        """Soft IoU reward using continuous sigmoid probabilities."""
        gt = (gt_mask > 0.5).float()
        intersection = (pred_prob * gt).sum()
        union = pred_prob.sum() + gt.sum() - intersection
        return float(intersection / (union + 1e-8))

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if not getattr(self.args, 'use_grpo', False):
            return super().training_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        # Initialize reference model if not done yet
        if not hasattr(self, 'ref_model'):
            try:
                self.ref_model = copy.deepcopy(model)
                self.ref_model.eval()
                for p in self.ref_model.parameters():
                    p.requires_grad = False
            except Exception as e:
                # Fallback if deepcopy is not supported (e.g. ZeRO-3 or OOM)
                self.ref_model = None

        device = inputs['images'].device
        batch_size = inputs['input_ids'].shape[0]
        
        # Accumulate and average the loss over the batch size to support batch_size > 1
        accumulated_loss = 0.0
        
        for b_idx in range(batch_size):
            input_ids = inputs['input_ids'][b_idx]
            labels = inputs['labels'][b_idx]
            
            # Find prompt length: index of first non-ignore token
            non_ignore_indices = (labels != -100).nonzero(as_tuple=True)[0]
            if len(non_ignore_indices) > 0:
                prompt_len = non_ignore_indices[0].item()
            else:
                prompt_len = len(labels)
                
            prompt_ids = input_ids[:prompt_len]
            image_tensor = inputs['images'][b_idx:b_idx+1] # [1, 3, H, W]
            gt_mask = inputs.get('masks', None)
            if gt_mask is not None:
                gt_mask = gt_mask[b_idx]

            # Replicate input prompt and images to batch size G before generating
            prompt_ids_batched = prompt_ids.unsqueeze(0).repeat(self.args.group_size, 1)
            images_batched = image_tensor.repeat(self.args.group_size, 1, 1, 1)

            # Replicate refer token ID and embedding indices to support generation/forward calls
            token_refer_id = inputs.get('token_refer_id', None)
            if token_refer_id is not None:
                token_refer_id_batched = [token_refer_id[b_idx]] * self.args.group_size
            else:
                token_refer_id_batched = None

            refer_embedding_indices = inputs.get('refer_embedding_indices', None)
            if refer_embedding_indices is not None:
                prompt_ref_indices = refer_embedding_indices[b_idx][:prompt_len]
                refer_embedding_indices_prompt_batched = prompt_ref_indices.unsqueeze(0).repeat(self.args.group_size, 1)
            else:
                refer_embedding_indices_prompt_batched = None

            # 1. Sample G candidate reasoning paths from active policy (no grads)
            attention_mask_batched = torch.ones_like(prompt_ids_batched)
            with self.compute_loss_context_manager():
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=prompt_ids_batched,
                        attention_mask=attention_mask_batched,
                        images=images_batched,
                        token_refer_id=token_refer_id_batched,
                        refer_embedding_indices=refer_embedding_indices_prompt_batched,
                        do_sample=True,
                        temperature=1.0,
                        return_dict_in_generate=True,
                    )

            output_ids = outputs.sequences  # [G, total_len]
            gen_ids = output_ids[:, prompt_ids.size(0):]  # [G, gen_len]
            gen_len = gen_ids.shape[1]

            # Clear CUDA cache to free up generation activations
            torch.cuda.empty_cache()

            # Filter out negative placeholder token IDs (like -200 and -204) before decoding
            clean_prompt_ids = prompt_ids[prompt_ids >= 0]
            question = self.tokenizer.decode(clean_prompt_ids, skip_special_tokens=True).replace(PREFIX_INST, "").strip()
            answer_texts = [self.tokenizer.decode(g[g >= 0], skip_special_tokens=True).strip() for g in gen_ids]

            image_cpu = image_tensor.squeeze(0).cpu()
            items = [_build_reasoning_item(image_cpu, question, ans, self.tokenizer) for ans in answer_texts]

            # Collate items using DataCollector
            batch = DataCollector(tokenizer=self.tokenizer)(items)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["token_refer_id"] = [t.to(device) for t in batch["token_refer_id"]]
            batch["token_answer_id"] = [t.to(device) for t in batch["token_answer_id"]]

            # Predict masks
            with self.compute_loss_context_manager():
                with torch.no_grad():
                    outputs_seg = model.eval_seg(
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
            # Clear CUDA cache after mask prediction
            torch.cuda.empty_cache()

            # Calculate rewards (IoU only, using soft IoU fallback to avoid early cold-start zero advantages)
            group_rewards = []
            for i, out in enumerate(outputs_seg):
                raw_mask = out.get("raw_masks", None)
                if raw_mask is not None:
                    pred_prob = raw_mask.sigmoid()
                    pred_prob = pred_prob[0] if pred_prob.dim() == 3 else pred_prob
                    pred_prob = pred_prob.to(device)
                    if gt_mask is not None:
                        r_iou = self.compute_soft_iou_reward(pred_prob, gt_mask)
                    else:
                        r_iou = 0.0
                else:
                    pred = out["pred_masks"]
                    pred_mask = pred[0] if pred.dim() == 3 else pred
                    if gt_mask is not None:
                        r_iou = self.compute_iou_reward(pred_mask, gt_mask)
                    else:
                        r_iou = 0.0
                group_rewards.append(r_iou)

            # Debug rewards logging
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float, device=device)
            print(f"[GRPO debug] rewards={['%.6f' % r for r in group_rewards]}  mean={rewards_tensor.mean().item():.6f}  std={rewards_tensor.std().item():.6f}")

            # Replicate refer embedding indices for the full output sequence (prompt + generation)
            if refer_embedding_indices is not None:
                gen_ref_indices = torch.zeros(gen_len, dtype=prompt_ref_indices.dtype, device=prompt_ref_indices.device)
                full_ref_indices = torch.cat([prompt_ref_indices, gen_ref_indices], dim=0)
                refer_embedding_indices_full_batched = full_ref_indices.unsqueeze(0).repeat(self.args.group_size, 1)
            else:
                refer_embedding_indices_full_batched = None

            with self.compute_loss_context_manager():
                # 3. Log probabilities under active policy (GRADIENTS ENABLED - for policy loss)
                new_logits = model(
                    input_ids=output_ids, 
                    images=images_batched,
                    token_refer_id=token_refer_id_batched,
                    refer_embedding_indices=refer_embedding_indices_full_batched
                ).logits
                new_logp = torch.nn.functional.log_softmax(new_logits, -1)[:, -gen_len-1:-1].gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
                old_logp = new_logp.detach()

                # 4. Log probabilities under reference policy (no gradient - for KL penalty)
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_outputs = self.ref_model(
                            input_ids=output_ids, 
                            images=images_batched,
                            token_refer_id=token_refer_id_batched,
                            refer_embedding_indices=refer_embedding_indices_full_batched
                        )
                    else:
                        # PEFT/LoRA adapter disabling fallback for DeepSpeed compatibility
                        if hasattr(model, 'disable_adapter'):
                            with model.disable_adapter():
                                ref_outputs = model(
                                    input_ids=output_ids, 
                                    images=images_batched,
                                    token_refer_id=token_refer_id_batched,
                                    refer_embedding_indices=refer_embedding_indices_full_batched
                                )
                        else:
                            ref_outputs = None

                    if ref_outputs is not None:
                        ref_log_probs = torch.nn.functional.log_softmax(ref_outputs.logits, dim=-1)
                        gen_slice = gen_ids.unsqueeze(-1)  # [G, gen_len, 1]
                        ref_token_log_probs = ref_log_probs[:, -gen_len-1:-1].gather(2, gen_slice).squeeze(-1)  # [G, gen_len]
                    else:
                        ref_token_log_probs = old_logp

            # 5. Calculate Group Relative Advantages (Local Normalization)
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float, device=device)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            advantages = ((rewards_tensor - mean_r) / std_r).unsqueeze(-1)  # [G, 1] to broadcast with gen_len

            # 6. Compute GRPO Loss (Policy Loss + stable k3 KL Penalty)
            # Ratio has active gradients through new_logp
            ratio = torch.exp(new_logp - old_logp)

            # Mask out padding tokens to prevent gradient distortion
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            gen_attention_mask = (gen_ids != pad_token_id).float()
            denom = gen_attention_mask.sum(dim=-1).clamp(min=1.0)

            # Policy objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = (-torch.min(surr1, surr2) * gen_attention_mask).sum(dim=-1) / denom  # [G]

            # Stable k3 KL Divergence Penalty: exp(ref_log_prob - log_prob) - 1 - (ref_log_prob - log_prob)
            log_ratio = ref_token_log_probs - new_logp
            kl = torch.exp(log_ratio) - 1 - log_ratio
            kl_loss = (self.args.kl_coeff * kl * gen_attention_mask).sum(dim=-1) / denom  # [G]

            # Total loss averaged over group size
            total_loss = (policy_loss + kl_loss).mean()
            accumulated_loss += total_loss

        # Explicit backward pass to update the active policy network
        loss_to_backward = accumulated_loss / batch_size

        if self.args.gradient_accumulation_steps > 1:
            loss_to_backward = loss_to_backward / self.args.gradient_accumulation_steps

        if self.use_apex:
            with amp.scale_loss(loss_to_backward, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss_to_backward)

        return loss_to_backward.detach()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()


    def compute_loss(self, model, inputs, return_outputs=False):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
                """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and 'loss_dice' in outputs:
                loss_dict = {}
                for name,value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self,'history_loss_dict'):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                # loss_mask = outputs["loss_mask"].item() if isinstance(outputs, dict) else 0
                # loss_dice = outputs["loss_dice"].item() if isinstance(outputs, dict) else 0
                # loss_SEG_class = outputs["loss_SEG_class"].item() if isinstance(outputs, dict) else 0
                # loss_class_name_class = outputs["loss_class_name_class"].item() if isinstance(outputs, dict) else 0
                # loss_dict = {
                #     'loss_mask':loss_mask,
                #     'loss_dice': loss_dice,
                #     'loss_SEG_class':loss_SEG_class,
                #     'loss_class_name_class': loss_class_name_class
                # }
                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss

    # def training_step(self, model, inputs) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)
    #
    #     return loss.detach() / self.args.gradient_accumulation_steps