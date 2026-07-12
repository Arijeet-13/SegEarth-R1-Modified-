import os
import sys
import torch

# Ensure SegEarth-R1 is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from segearth_r1.model.language_model.llava_phi import segearth_r1, LlavaConfig
from segearth_r1.train.train_dataset import get_mask_config
from segearth_r1.constants import IMAGE_TOKEN_INDEX, ANSWER_TOKEN_INDEX

def test_inference():
    print("1. Creating dummy model configuration...")
    config = LlavaConfig()
    
    # Configure dummy model sizes
    config.vocab_size = 1000
    config.hidden_size = 2048
    config.num_attention_heads = 32
    config.num_hidden_layers = 2  # Keep it small to avoid GPU memory limits
    config.swin_type = "base"
    config.mm_vision_tower = "dummy_swin"
    config.mask_decode_train = True
    config.mm_projector_type = "SparseConv_1"
    config.mm_hidden_size = 256
    config.projector_outdim = 2048
    config.with_norm = True
    config.with_layernorm = False
    config.attn_implementation = "eager"

    print("2. Loading mask decoder config...")
    # Load mask config from yaml
    mask_cfg = get_mask_config('./segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml')
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = 'referring'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"3. Instantiating model with random weights on {device}...")
    # Instantiate model
    model = segearth_r1(config, mask_decoder_cfg=mask_cfg, use_seg_query=False, use_multi_target_seg=True, seg_token_num=2).to(device)
    model.initial_mask_module()
    
    # Register special token ID for multi-target testing
    model.SEG_M_id = torch.tensor([999], dtype=torch.long, device=device)
    model.eval()
    print("[SUCCESS] Model initialized successfully!")

    print("4. Preparing dummy single-target inputs...")
    batch_size = 1
    seq_len = 10
    
    # Create input ids (dummy text tokens + 1 image token)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    input_ids[0, 3] = IMAGE_TOKEN_INDEX  # Place image token index
    
    attention_mask = torch.ones((batch_size, seq_len)).to(device)
    
    # Create random image tensor [1, 3, 1024, 1024]
    images = torch.randn(batch_size, 3, 1024, 1024).to(device)
    
    # Define dummy masks for the target region
    masks = [torch.randn(1, 1024, 1024).to(device) > 0]
    
    # Define reference token indices
    refer_embedding_indices = [torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.long).to(device)]

    print("5. Running single-target forward pass...")
    # Disable multi-target temporarily to check single-target behavior
    model.use_multi_target_seg = False
    with torch.no_grad():
        outputs_single = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            masks=masks,
            refer_embedding_indices=refer_embedding_indices,
            dataset_type=['refer_seg']
        )
    print("[SUCCESS] Single-target forward pass succeeded!")
    print(f"Single-target output keys: {outputs_single.keys()}")

    print("6. Preparing dummy multi-target inputs...")
    model.use_multi_target_seg = True
    
    # Create answer input sequence with multiple targets:
    # prompt <image> ... <answer_token> target1 <seg_m> <seg_m> target2 <seg_m> <seg_m>
    # Token IDs: image_token at 3, answer_token at 5, seg_m (25000) at 7, 8, 10, 11
    input_ids = torch.randint(0, config.vocab_size, (batch_size, 15)).to(device)
    input_ids[0, 3] = IMAGE_TOKEN_INDEX
    input_ids[0, 5] = ANSWER_TOKEN_INDEX
    input_ids[0, 7] = 999
    input_ids[0, 8] = 999
    input_ids[0, 10] = 999
    input_ids[0, 11] = 999
    
    attention_mask = torch.ones((batch_size, 15)).to(device)
    
    # Build list of list of masks: [[mask_target1, mask_target2]]
    masks_multi = [[torch.randn(1, 1024, 1024).to(device) > 0, torch.randn(1, 1024, 1024).to(device) > 0]]
    
    # token_answer_id contains the answer tokens (length 9) starting from index 6 (after ANSWER_TOKEN_INDEX at index 5)
    token_answer_id = [input_ids[0, 6:].clone()]
    
    # target_group_ids corresponding to token_answer_id (length 9)
    target_group_ids = [torch.tensor([0, 1, 1, 0, 2, 2, 0, 0, 0], dtype=torch.long).to(device)]
    
    answer_embedding_indices = [torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long).to(device)]
    refer_embedding_indices = [torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long).to(device)]

    print("7. Running multi-target forward pass...")
    labels = input_ids.clone()
    with torch.no_grad():
        outputs_multi = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            masks=masks_multi,
            labels=labels,
            token_answer_id=token_answer_id,
            target_group_ids=target_group_ids,
            answer_embedding_indices=answer_embedding_indices,
            refer_embedding_indices=refer_embedding_indices,
            dataset_type=['reason_seg']
        )
    print("[SUCCESS] Multi-target forward pass succeeded!")
    print(f"Multi-target output keys: {outputs_multi.keys()}")

    print("8. Running multi-target eval_seg pass (with target_group_ids)...")
    with torch.no_grad():
        eval_outputs = model.eval_seg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            masks=masks_multi,
            token_answer_id=token_answer_id,
            target_group_ids=target_group_ids,
            answer_embedding_indices=answer_embedding_indices,
            refer_embedding_indices=refer_embedding_indices
        )
    print("[SUCCESS] Multi-target eval_seg pass succeeded!")
    print(f"First element in eval_outputs: {eval_outputs[0].keys()}")
    
    print("9. Running multi-target eval_seg pass (WITHOUT target_group_ids - dynamic construction test)...")
    with torch.no_grad():
        eval_outputs_dynamic = model.eval_seg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            masks=masks_multi,
            token_answer_id=token_answer_id,
            target_group_ids=None,
            answer_embedding_indices=answer_embedding_indices,
            refer_embedding_indices=refer_embedding_indices
        )
    print("[SUCCESS] Multi-target eval_seg pass with dynamic target_group_ids succeeded!")
    print(f"First element in dynamic eval_outputs: {eval_outputs_dynamic[0].keys()}")

if __name__ == "__main__":
    test_inference()
