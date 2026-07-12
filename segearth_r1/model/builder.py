#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import warnings
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from segearth_r1.model import *
from segearth_r1.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from segearth_r1.train.train_dataset import get_mask_config
from segearth_r1.model.language_model.llava_phi import segearth_r1

def load_pretrained_model(model_path, model_base, model_name, model_args, mask_config='./segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml', use_seg_query = False, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": 'cpu'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    print('loading segmentation model')

    model_map_name = model_args.model_map_name
    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, 'seg_task') else 'instance'

    use_multi_target_seg = getattr(model_args, 'use_multi_target_seg', False) if model_args is not None else False
    seg_token_num = getattr(model_args, 'seg_token_num', 1) if model_args is not None else 1

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if "[SEG]" not in tokenizer.get_vocab():
        tokenizer.add_tokens("[SEG]")
    
    if use_multi_target_seg and "<seg_m>" not in tokenizer.get_vocab():
        tokenizer.add_tokens("<seg_m>")

    print(f'current model is {model_map_name}')
    model = segearth_r1.from_pretrained(
        model_path,
        mask_decoder_cfg=mask_cfg,
        use_seg_query=use_seg_query,
        use_multi_target_seg=use_multi_target_seg,
        seg_token_num=seg_token_num,
        **kwargs
    )
    model.resize_token_embeddings(len(tokenizer))
    
    seg_m_kwargs = {}
    if use_multi_target_seg:
        seg_m_kwargs['SEG_M'] = tokenizer("<seg_m>", return_tensors='pt', add_special_tokens=False)['input_ids']
    model.get_special_token(
        SEG=tokenizer("[SEG]", return_tensors='pt', add_special_tokens=False)['input_ids'],
        EOS=tokenizer.eos_token_id,
        **seg_m_kwargs
    )

    vision_tower = model.get_vision_tower()
        
    vision_tower.to(device=device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
