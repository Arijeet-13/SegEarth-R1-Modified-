import sys
sys.path.append('')
import os
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import warnings
import pathlib
from typing import Dict, Optional, Sequence, List
import cv2
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer

from segearth_r1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, \
    REFER_TOKEN_INDEX,ANSWER_TOKEN_INDEX, DEFAULT_SEG_M_TOKEN
from torch.utils.data import Dataset
from segearth_r1 import conversation as conversation_lib
from segearth_r1.model import *
from segearth_r1.train.refer import REFER
from segearth_r1.mask_config.config import Config
from fvcore.common.config import CfgNode

def get_mask_config(config='./segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg

def preprocess_mask(mask, image_size):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    
    bs, h, w = mask.shape
    
    processed_masks = []
    
    for i in range(bs):
        single_mask = mask[i]
        
        hh, ww = single_mask.shape[:2]
        
        if ww > hh:
            new_w = image_size
            new_h = int(hh * (image_size / ww))
        else:
            new_h = image_size
            new_w = int(ww * (image_size / hh))
        
        resized_mask = cv2.resize(single_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        pad_h = image_size - new_h
        pad_w = image_size - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded_mask = cv2.copyMakeBorder(resized_mask, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=0)
        
        processed_masks.append(padded_mask)
    
    processed_masks = np.stack(processed_masks, axis=0)
    processed_masks = torch.from_numpy(processed_masks).to(torch.uint8)
    
    return processed_masks

def preprocess_image(image, image_size, pad_value=0):
    
    h, w = image.shape[:2]
    
    if w > h:
        new_w = image_size
        new_h = int(h * (image_size / w))
    else:
        new_h = image_size
        new_w = int(w * (image_size / h))
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=pad_value)
    
    padded_image = padded_image.transpose(2,0,1)
    return padded_image

def tokenizer_special_tokens(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX ,refer_token_index=REFER_TOKEN_INDEX, answer_token_index = ANSWER_TOKEN_INDEX, return_tensors=None):
    input_ids = []
    special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<refer>':refer_token_index, '<answer>':answer_token_index}
    prompt_chunks = re.split('(<image>|<seg>|<refer>|<answer>)', prompt)

    for chunk in prompt_chunks:
        if chunk in special_token_map:
            input_ids.append(special_token_map[chunk])
        else:
            input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long).squeeze()
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    else:
        return input_ids

def preprocess_llama2(sources, tokenizer):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    input_ids = torch.stack(
        [tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer_special_tokens(rou, tokenizer))
            instruction_len = len(tokenizer_special_tokens(parts[0], tokenizer)) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

        
def preprocess_referring_instruction(instruction, tokenizer, REFER_token='[SEG]'):
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    tokenized = tokenized + [tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

    token_refer_id = torch.tensor(tokenized)

    return token_refer_id

class RefSegRSDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, image_size = 1024):
        super(RefSegRSDataset, self).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.RefSegRS_images_root = os.path.join(self.base_data_path, "rs_ref_seg/RefSegRS/images")
        self.RefSegRS_labels_root = os.path.join(self.base_data_path, "rs_ref_seg/RefSegRS/masks")
        self.RefSegRS_txt = os.path.join(self.base_data_path, "rs_ref_seg/RefSegRS/output_phrase_train.txt")       
        with open(self.RefSegRS_txt, 'r') as file:
            phases = file.readlines()
        images = []
        masks = []
        refs = []
        for phase in phases:
            match = re.match(r'(\d+)\s+(.*)', phase.strip())
            if match:
                image_path = os.path.join(self.RefSegRS_images_root, match.group(1)+'.tif')
                label_path = os.path.join(self.RefSegRS_labels_root, match.group(1)+'.tif')
                ref = match.group(2)
                images.append(image_path)
                masks.append(label_path)
                refs.append(ref)
        self.images = images
        self.masks = masks
        self.refs = refs

    def __len__(self):
        return len(self.images)    

    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        
        ref = self.refs[idx]
        label_path = self.masks[idx]
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 1
        processed_mask = preprocess_mask(mask, self.image_size)
        data_dict['mask'] = processed_mask
        
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ' {}'.format(ref)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure. It is <seg>. '}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'refer_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        
        return data_dict

class ReasonSegDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, image_size = 1024):
        super(ReasonSegDataset, self).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.ReasonSeg_images_root = os.path.join(base_data_path, "rs_reason_seg/RSReasonSeg/train/images")
        self.ReasonSeg_labels_root = os.path.join(base_data_path, "rs_reason_seg/RSReasonSeg/train/labels")
        self.ReasonSeg_QAs_root = os.path.join(base_data_path, "rs_reason_seg/RSReasonSeg/train/QAs")
        self.images = self.load_file_paths(self.ReasonSeg_images_root, valid_extensions=('.jpg', '.jpeg', '.png'))
        self.labels = self.load_file_paths(self.ReasonSeg_labels_root, valid_extensions=('.png',))
        self.QAs_paths = self.load_file_paths(self.ReasonSeg_QAs_root, valid_extensions=('.json', '.txt'))
        self.QAs = []
        for QAs_path in self.QAs_paths:
            with open(QAs_path, "r") as file:
                QA = json.load(file)
            self.QAs.append(QA)
    
    def __len__(self):
        return len(self.images)
    
    def load_file_paths(self, directory, valid_extensions=None):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        file_paths = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if valid_extensions is None or filename.lower().endswith(valid_extensions):
                    file_paths.append(file_path)
        
        file_paths.sort()
        print(f"Found {len(file_paths)} files in {directory}")
        return file_paths

    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.images[idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        
        label_path = self.labels[idx]
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        processed_mask = preprocess_mask(mask, self.image_size)
        data_dict['mask'] = processed_mask
        
        QAs = self.QAs[idx]
        
        question_num = len(QAs["questions"])
        question = QAs["questions"][0]
        
        answer_num = len(QAs["answer"])
        if answer_num == 0:
            answer = "There is no target object in the image."
        else:
            # answer_idx = random.randint(0, answer_num - 1)
            answer = QAs["answer"][0]
        
        prefix_inst = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
        instruction = ' {}'.format(question)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': 'Sure, It is <seg>. \n<answer>.'}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'reason_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        token_answer_id = preprocess_referring_instruction(answer, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        answer_embedding_indices = torch.zeros_like(input_ids)
        answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1
        
        data_dict['token_refer_id'] = token_refer_id
        data_dict['token_answer_id'] = token_answer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['answer_embedding_indices'] = answer_embedding_indices

        return data_dict

class RRSISDDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, image_size = 1024):
        super(RRSISDDataset, self).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.RRSISD_data_root = os.path.join(base_data_path, "rs_ref_seg/RRSIS-D/")
        refer = REFER(self.RRSISD_data_root,'rrsisd','unc')
        ref_ids = refer.getRefIds(split='train')
        all_imgs = refer.Imgs
        imgs = list(all_imgs[i] for i in ref_ids)
        images =[]
        for img in imgs:
            image_path = os.path.join(self.RRSISD_data_root, 'images/rrsisd/JPEGImages', img['file_name'])
            images.append(image_path)
        refs = []
        for i, img_id in enumerate(ref_ids):
            ref = refer.Refs[img_id]
            refs.append(ref['sentences'][0]['raw'])
        masks = []
        for i in ref_ids:
            mask = refer.getMask(i)
            masks.append(mask)
        self.images = images
        self.refs = refs
        self.masks = masks
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        
        ref = self.refs[idx]
        mask = self.masks[idx]
        processed_mask = preprocess_mask(mask, self.image_size)
        data_dict['mask'] = processed_mask
        
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ' {}'.format(ref)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure. It is <seg>. '}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'refer_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        
        return data_dict
    
def preprocess_multimodal(
        sources,
        data_args
):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources
 
class MM_Conv_Dataset(Dataset):
    def __init__(self, base_data_path, tokenizer, data_args, image_size = 1024):
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.data_args = data_args
        json_path = os.path.join(base_data_path, "rs_vqa/FIT-RS/FIT-RS-train-sampled-381k(subset of 1415k).json")
        with open(json_path, 'r') as data:
            self.list_data_dict = json.load(data)
    def __len__(self):
        return len(self.list_data_dict)
    
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    def __getitem__(self, idx):
        sources = self.list_data_dict[idx]
        sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = {}
        if 'image' in sources[0]:
            image_path = os.path.join(self.base_data_path, "rs_vqa/FIT-RS/FIT-RS_Img/imgv2_split_512_100_vaild/" + sources[0]['image'])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = preprocess_image(image, self.image_size)
            processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
            data_dict['image'] = processed_image
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        text_dict = preprocess_llama2(sources, self.tokenizer) 
        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'mm_conv'
        if 'image' not in data_dict:
            # image does not exist in the data, but the model is multimodal
            data_dict['image'] = torch.zeros(3, self.image_size, self.image_size)
        return data_dict  



def preprocess_multi_target_answer(target_texts, tokenizer, seg_token_num, seg_m_token=DEFAULT_SEG_M_TOKEN):
    """
    Build one answer token sequence covering multiple targets, each followed by
    `seg_token_num` consecutive [SEG_M] tokens, plus a parallel `target_group_ids`
    tensor (0 = ordinary text, k = the k-th [SEG_M] group, 1-indexed) so the model
    can later gather/fuse each target's tokens separately. Fully separate from
    preprocess_referring_instruction (which drives the existing binary/[SEG] path).
    """
    seg_m_id = tokenizer.encode(seg_m_token, add_special_tokens=False)[0]
    all_ids, all_group_ids = [], []
    for target_idx, text in enumerate(target_texts, start=1):
        word_ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(word_ids)
        all_group_ids.extend([0] * len(word_ids))
        all_ids.extend([seg_m_id] * seg_token_num)
        all_group_ids.extend([target_idx] * seg_token_num)
    token_answer_id = torch.tensor(all_ids, dtype=torch.long)
    target_group_ids = torch.tensor(all_group_ids, dtype=torch.long)
    return token_answer_id, target_group_ids


class MultiTargetReasonSegDataset(Dataset):
    """
    PixelLM-style multi-target wrapper around ReasonSegDataset. Does NOT modify
    ReasonSegDataset or its data files - composes over it read-only, so the
    binary/single-target training path is completely unaffected whether or not
    this class is ever used.

    NOTE on data assumption: RSReasonSeg ships one binary mask per image (verified:
    images/labels/QAs are 1:1). There is no native multi-instance annotation to
    draw on, so multiple targets per training example are synthesized by running
    connected-component analysis on that single binary mask. If your data
    actually has real multi-instance annotations, swap `_build_targets` below to
    read them instead - everything downstream (token building, collator, model)
    is agnostic to where the per-instance masks come from.
    """
    def __init__(self, base_data_path, tokenizer, image_size=1024, seg_token_num=1,
                 max_targets_per_sample=3, min_component_area=100):
        super().__init__()
        self.base = ReasonSegDataset(base_data_path, tokenizer, image_size)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.seg_token_num = max(1, seg_token_num)
        self.max_targets_per_sample = max_targets_per_sample
        self.min_component_area = min_component_area

    def __len__(self):
        return len(self.base)

    def _build_targets(self, raw_mask):
        """Split one binary mask into up to N per-instance masks via connected components."""
        raw_mask = (raw_mask != 0).astype(np.uint8)
        num_labels, components = cv2.connectedComponents(raw_mask, connectivity=8)
        instance_masks = []
        for label in range(1, num_labels):  # 0 = background
            inst = (components == label).astype(np.uint8)
            if inst.sum() >= self.min_component_area:
                instance_masks.append(inst)
        if len(instance_masks) == 0:
            # No component survived the area filter (or mask was empty) - fall back
            # to the original single mask so this sample is still usable.
            instance_masks = [raw_mask]
        # Largest components first, capped at max_targets_per_sample
        instance_masks.sort(key=lambda m: m.sum(), reverse=True)
        return instance_masks[:self.max_targets_per_sample]

    def __getitem__(self, idx):
        data_dict = {}
        image_path = self.base.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        processed_image = (torch.tensor(processed_image) - self.base.pixel_mean) / self.base.pixel_std
        data_dict['image'] = processed_image

        label_path = self.base.labels[idx]
        raw_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        instance_masks = self._build_targets(raw_mask)
        num_targets = len(instance_masks)
        data_dict['masks'] = [preprocess_mask(m, self.image_size) for m in instance_masks]

        QAs = self.base.QAs[idx]
        question = QAs["questions"][0]
        answer_num = len(QAs["answer"])
        base_answer = QAs["answer"][0] if answer_num > 0 else "There is no target object in the image."

        prefix_inst = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
        instruction = ' {}'.format(question)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': 'Sure, It is <seg>. \n<answer>.'}]]
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'reason_seg'  # reuse existing branch, no forward() changes needed

        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        # One text segment per synthesized target, each capped by seg_token_num [SEG_M] tokens
        target_texts = [f"{base_answer}, instance {i + 1}" for i in range(num_targets)]
        token_answer_id, target_group_ids = preprocess_multi_target_answer(
            target_texts, self.tokenizer, self.seg_token_num)

        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        answer_embedding_indices = torch.zeros_like(input_ids)
        answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1

        data_dict['token_refer_id'] = token_refer_id
        data_dict['token_answer_id'] = token_answer_id
        data_dict['target_group_ids'] = target_group_ids  # same length as token_answer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['answer_embedding_indices'] = answer_embedding_indices
        return data_dict


class MultiTargetDataCollector(object):
    """
    Separate collator for MultiTargetReasonSegDataset - does not touch/replace
    DataCollector, so the binary-segmentation dataloader is unaffected.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data_dicts):
        input_ids = [d['input_ids'] for d in data_dicts]
        labels = [d['labels'] for d in data_dicts]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        batch['images'] = torch.stack([d['image'] for d in data_dicts])
        # List of lists: masks[i] is a list of per-target masks for sample i.
        # forward() already branches on isinstance(masks[0], (list, tuple)).
        batch['masks'] = [d['masks'] for d in data_dicts]
        batch['dataset_type'] = [d['dataset_type'] for d in data_dicts]
        batch['token_refer_id'] = [d['token_refer_id'] for d in data_dicts]
        # Kept as plain lists (not padded) so token_answer_id[i] and
        # target_group_ids[i] stay length-aligned per sample.
        batch['token_answer_id'] = [d['token_answer_id'] for d in data_dicts]
        batch['target_group_ids'] = [d['target_group_ids'] for d in data_dicts]

        refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
            [d['refer_embedding_indices'] for d in data_dicts], batch_first=True, padding_value=0)
        batch['refer_embedding_indices'] = refer_embedding_indices
        answer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
            [d['answer_embedding_indices'] for d in data_dicts], batch_first=True, padding_value=0)
        batch['answer_embedding_indices'] = answer_embedding_indices
        return batch


class DataCollector(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data_dicts):
        input_ids = [data_dict['input_ids'] for data_dict in data_dicts]
        labels = [data_dict['labels'] for data_dict in data_dicts]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),    
        )
        
        if 'image' in data_dicts[0]:
            images = [data_dict['image'] for data_dict in data_dicts]
            batch['images'] = torch.stack(images)
        if 'mask' in data_dicts[0]:
            masks = [data_dict['mask'] for data_dict in data_dicts]
            batch['masks'] = torch.stack(masks)
        
        for data_dict in data_dicts:
            for key in ['input_ids', 'labels', 'image']:
                del data_dict[key]

        if 'dataset_type' in data_dicts[0]:
            batch['dataset_type'] = [data_dict['dataset_type'] for data_dict in data_dicts]
            
        if 'token_refer_id' in data_dicts[0]:
            token_refer_id = [data_dict['token_refer_id'] for data_dict in data_dicts]
            batch['token_refer_id'] = token_refer_id
        
        if 'token_answer_id' in data_dicts[0]:
            token_answer_id = [data_dict['token_answer_id'] for data_dict in data_dicts]
            batch['token_answer_id'] = token_answer_id
            
        
        if 'refer_embedding_indices' in data_dicts[0]:
            refer_embedding_indices = [data_dict['refer_embedding_indices'] for data_dict in data_dicts]
            refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                refer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['refer_embedding_indices'] = refer_embedding_indices
            
        if 'answer_embedding_indices' in data_dicts[0]:
            answer_embedding_indices = [data_dict['answer_embedding_indices'] for data_dict in data_dicts]
            answer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                answer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['answer_embedding_indices'] = answer_embedding_indices
            
        return batch
    
if __name__ == "__main__":
    
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_phi']
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = RRSISDDataset(
    base_data_path = "/root/siton-data-412581749c3f4cfea0d7c972b8742057/data", 
    tokenizer = tokenizer,
    image_size =1024,
    )
    
    data = dataset[0]
    print(data['image'].shape)