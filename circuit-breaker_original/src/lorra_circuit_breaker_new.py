from functools import partial
import logging
import os
import json
import gc
import atexit
import numpy as np

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
import transformers
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.integrations import deepspeed
import torch

# from cb_train_dataset import (
#     CircuitBreakerDataset
# )

# from train_datasets_original import (
#     CircuitBreakerDataset
# )

from cb_train_dataset_infinity import (
    InfinityCircuitBreakerDataset
)

# =============== INFINITY ADAPTATION IMPORTS ===============
# Import Infinity-specific modules for T2I adaptation
import sys
sys.path.append('../infinity')
try:
    from infinity.models.infinity import Infinity
    from infinity.utils import arg_util, misc, wandb_utils
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    INFINITY_AVAILABLE = True
except ImportError:
    INFINITY_AVAILABLE = False
    print("Warning: Infinity modules not available, T2I features disabled")

from utils import save_model_and_tokenizer
from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import gc

def compute_infinity_circuit_breaker_loss(
    trainer, 
    model, 
    inputs: Dict,
    target_layers: List[int] = [0, 1, 2, 3, 4, 5],
    alpha: float = 0.1,
    return_outputs: bool = False,
    tokenizer = None
):
    """
    Compute circuit breaker loss for Infinity models
    
    Args:
        trainer: InfinityTrainer instance
        model: The model to compute loss for
        inputs: Dictionary containing:
            - text_cond_tuple: retain text condition tuple
            - text_cond_tuple_circuit_breaker: circuit breaker text condition tuple  
            - text_cond_tuple_val: validation text condition tuple
            - input_ids: retain input ids
            - attention_mask: retain attention mask
            - input_ids_circuit_breaker: cb input ids
            - attention_mask_circuit_breaker: cb attention mask
            - input_ids_val: validation input ids
            - attention_mask_val: validation attention mask
        target_layers: List of layer indices to apply circuit breaker to
        alpha: Circuit breaker coefficient
        return_outputs: Whether to return outputs
        tokenizer: Tokenizer for debugging
    
    Returns:
        Combined loss (retain_loss + circuit_breaker_loss)
    """
    
    # Extract inputs
    retain_text_cond = inputs.get('text_cond_tuple')
    cb_text_cond = inputs.get('text_cond_tuple_circuit_breaker') 
    val_text_cond = inputs.get('text_cond_tuple_val')
    
    retain_input_ids = inputs.get('input_ids')
    retain_attention_mask = inputs.get('attention_mask')
    cb_input_ids = inputs.get('input_ids_circuit_breaker')
    cb_attention_mask = inputs.get('attention_mask_circuit_breaker')
    val_input_ids = inputs.get('input_ids_val')
    val_attention_mask = inputs.get('attention_mask_val')
    
    # Get training progress for scheduling
    progress = trainer.prog_it / trainer.max_it if hasattr(trainer, 'max_it') else 0.5
    scheduled_coeff = progress
    retain_coeff = alpha * scheduled_coeff
    circuit_breaker_coeff = alpha * (1 - scheduled_coeff)
    
    print(f'\nPROGRESS: {progress:.4f}', '='*50)
    print(f"retain_coeff: {retain_coeff:.4f} || circuit_breaker_coeff: {circuit_breaker_coeff:.4f}")
    
    # Initialize losses
    retain_loss = torch.tensor(0.0, device=model.device)
    circuit_breaker_loss = torch.tensor(0.0, device=model.device)
    
    # ===== RETAIN LOSS =====
    if retain_coeff > 0 and retain_text_cond is not None:
        # Get original model outputs (frozen)
        with torch.no_grad():
            model.eval()
            # Create dummy image tokens for retain (since we only have text)
            dummy_image_tokens = torch.randn(1, 256, 768, device=model.device)  # Adjust size as needed
            
            # Get original hidden states - try different model structures
            orig_outputs = model(retain_text_cond, dummy_image_tokens, scale_schedule=[(1, 16, 16)])
            orig_hidden_states = []
            for layer_idx in target_layers:
                if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
                    if hasattr(model.blocks[layer_idx], 'output_hidden_states'):
                        orig_hidden_states.append(model.blocks[layer_idx].output_hidden_states.detach())
                elif hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                    if hasattr(model.unregistered_blocks[layer_idx], 'get_hidden_states'):
                        orig_hidden_states.append(model.unregistered_blocks[layer_idx].get_hidden_states().detach())
            
            model.train()
        
        # Get current model outputs
        current_outputs = model(retain_text_cond, dummy_image_tokens, scale_schedule=[(1, 16, 16)])
        current_hidden_states = []
        for layer_idx in target_layers:
            if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
                if hasattr(model.blocks[layer_idx], 'output_hidden_states'):
                    current_hidden_states.append(model.blocks[layer_idx].output_hidden_states)
            elif hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                if hasattr(model.unregistered_blocks[layer_idx], 'get_hidden_states'):
                    current_hidden_states.append(model.unregistered_blocks[layer_idx].get_hidden_states())
        
        # Compute L2 distance loss
        if len(orig_hidden_states) > 0 and len(current_hidden_states) > 0:
            orig_stacked = torch.stack(orig_hidden_states)
            current_stacked = torch.stack(current_hidden_states)
            
            # Apply attention mask if available
            if retain_attention_mask is not None:
                mask = retain_attention_mask.unsqueeze(-1).expand_as(orig_stacked)
                orig_stacked = orig_stacked * mask
                current_stacked = current_stacked * mask
            
            retain_loss = torch.norm(current_stacked - orig_stacked, dim=-1, p=2).nanmean()
            
            if retain_coeff > 0:
                retain_cosine = F.cosine_similarity(current_stacked, orig_stacked, dim=-1)
                if retain_attention_mask is not None:
                    retain_cosine = retain_cosine * retain_attention_mask
                print(f"\nretain_cos_sim: {retain_cosine.mean().item():.4f}")
    
    # ===== CIRCUIT BREAKER LOSS =====
    if circuit_breaker_coeff > 0 and cb_text_cond is not None:
        # Get original model outputs (frozen)
        with torch.no_grad():
            model.eval()
            # Create dummy image tokens for circuit breaker
            dummy_image_tokens = torch.randn(1, 256, 768, device=model.device)  # Adjust size as needed
            
            # Get original hidden states - try different model structures
            orig_cb_outputs = model(cb_text_cond, dummy_image_tokens, scale_schedule=[(1, 16, 16)])
            orig_cb_hidden_states = []
            for layer_idx in target_layers:
                if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
                    if hasattr(model.blocks[layer_idx], 'output_hidden_states'):
                        orig_cb_hidden_states.append(model.blocks[layer_idx].output_hidden_states.detach())
                elif hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                    if hasattr(model.unregistered_blocks[layer_idx], 'get_hidden_states'):
                        orig_cb_hidden_states.append(model.unregistered_blocks[layer_idx].get_hidden_states().detach())
            
            model.train()
        
        # Get current model outputs
        current_cb_outputs = model(cb_text_cond, dummy_image_tokens, scale_schedule=[(1, 16, 16)])
        current_cb_hidden_states = []
        for layer_idx in target_layers:
            if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
                if hasattr(model.blocks[layer_idx], 'output_hidden_states'):
                    current_cb_hidden_states.append(model.blocks[layer_idx].output_hidden_states)
            elif hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                if hasattr(model.unregistered_blocks[layer_idx], 'get_hidden_states'):
                    current_cb_hidden_states.append(model.unregistered_blocks[layer_idx].get_hidden_states())
        
        # Compute inner product loss (maximize dissimilarity)
        if len(orig_cb_hidden_states) > 0 and len(current_cb_hidden_states) > 0:
            orig_cb_stacked = torch.stack(orig_cb_hidden_states)
            current_cb_stacked = torch.stack(current_cb_hidden_states)
            
            # Normalize vectors
            normalized_current = current_cb_stacked / (torch.norm(current_cb_stacked, dim=-1, keepdim=True) + 1e-8)
            normalized_orig = orig_cb_stacked / (torch.norm(orig_cb_stacked, dim=-1, keepdim=True) + 1e-8)
            
            # Compute inner product
            inner_product = (normalized_current * normalized_orig).sum(dim=-1)
            
            # Apply attention mask if available
            if cb_attention_mask is not None:
                mask = cb_attention_mask.unsqueeze(-1).expand_as(inner_product)
                inner_product = inner_product * mask
                mask_sum = mask.sum()
            else:
                mask_sum = inner_product.numel()
            
            # Circuit breaker loss: minimize similarity (maximize dissimilarity)
            circuit_breaker_loss = torch.relu(inner_product.sum()) / (mask_sum + 1e-8)
            
            if circuit_breaker_coeff > 0:
                print(f"\nupdated_cb_activations_norm: {current_cb_stacked.norm(dim=-1).mean().item():.4f}")
                print(f"orig_cb_activations_norm: {orig_cb_stacked.norm(dim=-1).mean().item():.4f}")
                
                cb_cosine = F.cosine_similarity(current_cb_stacked, orig_cb_stacked, dim=-1)
                if cb_attention_mask is not None:
                    cb_cosine = cb_cosine * cb_attention_mask
                print(f"cb_cos_sim: {cb_cosine.mean().item():.4f}")
    
    # ===== VALIDATION OBSERVATION =====
    if val_text_cond is not None:
        with torch.no_grad():
            model.eval()
            dummy_image_tokens = torch.randn(1, 256, 768, device=model.device)
            
            # Get both original and current outputs for validation
            orig_val_outputs = model(val_text_cond, dummy_image_tokens, scale_schedule=[(1, 16, 16)])
            current_val_outputs = model(val_text_cond, dummy_image_tokens, scale_schedule=[(1, 16, 16)])
            
            orig_val_hidden_states = []
            current_val_hidden_states = []
            for layer_idx in target_layers:
                if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
                    if hasattr(model.blocks[layer_idx], 'output_hidden_states'):
                        orig_val_hidden_states.append(model.blocks[layer_idx].output_hidden_states.detach())
                        current_val_hidden_states.append(model.blocks[layer_idx].output_hidden_states.detach())
                elif hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                    if hasattr(model.unregistered_blocks[layer_idx], 'get_hidden_states'):
                        orig_val_hidden_states.append(model.unregistered_blocks[layer_idx].get_hidden_states().detach())
                        current_val_hidden_states.append(model.unregistered_blocks[layer_idx].get_hidden_states().detach())
            
            model.train()
            
            if len(orig_val_hidden_states) > 0 and len(current_val_hidden_states) > 0:
                orig_val_stacked = torch.stack(orig_val_hidden_states)
                current_val_stacked = torch.stack(current_val_hidden_states)
                
                val_cosine = F.cosine_similarity(current_val_stacked, orig_val_stacked, dim=-1)
                if val_attention_mask is not None:
                    val_cosine = val_cosine * val_attention_mask
                print(f"val_cos_sim: {val_cosine.mean().item():.4f}")
    
    # Combine losses
    total_loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss
    
    print(f"\nretain_loss: {retain_loss.item():.4f}")
    print(f"circuit_breaker_loss: {circuit_breaker_loss.item():.4f}")
    print('='*50)
    
    if return_outputs:
        return (total_loss, retain_loss, circuit_breaker_loss)
    else:
        return total_loss


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def get_infinity_generation(text_prompt, model, tokenizer, vae, scale_schedule, device='cuda'):
    """
    INFINITY T2I Generation Function
    Key differences from LLM version:
    1. Input: text prompt instead of chat template
    2. Process: text encoding -> image token generation -> VAE decoding
    3. Output: image instead of text
    """
    # Tokenize text prompt
    tokens = tokenizer(text_prompt, max_length=tokenizer.model_max_length, 
                      padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    
    # Encode text using T5
    from transformers import T5EncoderModel
    text_encoder = T5EncoderModel.from_pretrained('t5-base').to(device)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    
    # Prepare text condition tuple
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    
    # Generate image
    with torch.no_grad():
        # Start with random tokens or use VAE to encode a starting image
        start_tokens = torch.randint(0, model.V, (1, 256), device=device)  # Simplified
        
        generated_tokens = model.autoregressive_infer_cfg(
            label_B_or_BLT=text_cond_tuple,
            scale_schedule=scale_schedule,
            vae=vae,
            B=1
        )
        
        # Decode tokens to image using VAE
        # This is a simplified version - actual implementation may vary
        generated_image = vae.decode(generated_tokens)
        
    return generated_image


def data_collator_infinity(batch_list):
    """
    Data collator for Infinity circuit breaker training
    """
    batch_inputs = {}
    
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k, []).append(input)
    
    # Stack tensors
    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.stack(inputs)
        elif isinstance(inputs[0], tuple):
            # Handle text condition tuples specially
            if k in ['text_cond_tuple', 'text_cond_tuple_circuit_breaker', 'text_cond_tuple_val']:
                # Stack the first element (kv_compact) and keep others as lists
                kv_compact_list = [item[0] for item in inputs]
                lens_list = [item[1] for item in inputs]
                cu_seqlens_k_list = [item[2] for item in inputs]
                Ltext_list = [item[3] for item in inputs]
                
                # Concatenate kv_compact along batch dimension
                kv_compact = torch.cat(kv_compact_list, dim=0)
                batch_inputs[k] = (kv_compact, lens_list, cu_seqlens_k_list, Ltext_list)
    
    return batch_inputs

def train():
    """
    Main training function for Infinity T2I Circuit Breaker - Selective Layer Fine-tuning
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    print(lorra_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)

    model_name_or_path = model_args.model_name_or_path
    target_layers = lorra_args.target_layers
    transform_layers = lorra_args.transform_layers
    full_layers = lorra_args.full_layers

    lorra_target_layers = [int(layer) for layer in target_layers.split(",")]
    
    # =============== INFINITY MODEL LOADING ===============
    # Load Infinity model components using the same approach as train.py
    from infinity.utils.load import build_vae_gpt
    from infinity.utils import arg_util
    
    # Create args for Infinity model (similar to train.py)
    args = arg_util.Args()
    args.model = getattr(model_args, 'model', "infinity_2b")
    args.vae_ckpt = getattr(model_args, 'vae_ckpt', "weights/infinity_vae_d32_rdn_short.pth")
    args.device = "cuda"
    args.model_init_device = "cuda"
    
    # Build VAE and GPT models (same as train.py)
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu') if os.path.exists(args.vae_ckpt) else {}
    vae_local, gpt_wo_ddp, gpt_wo_ddp_ema = build_vae_gpt(args, vae_ckpt, skip_gpt=False, device=args.model_init_device)
    
    # Load pretrained weights if specified (same as train.py with rush_resume)
    if model_name_or_path and os.path.exists(model_name_or_path):
        print(f"Loading pretrained weights from {model_name_or_path}")
        cpu_d = torch.load(model_name_or_path, 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
        else:
            state_dict = cpu_d
        
        def drop_unfit_weights(state_dict):
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                del state_dict['head.bias']
            return state_dict
        
        gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False)
    
    # =============== SELECTIVE LAYER FINE-TUNING ===============
    # Freeze all parameters first
    for param in gpt_wo_ddp.parameters():
        param.requires_grad = False
    
    # Unfreeze only specific layers for fine-tuning
    selective_layers = getattr(model_args, 'selective_layers', "0,1,2,3,4,5")
    if selective_layers != "all":
        selective_layer_indices = [int(layer) for layer in selective_layers.split(",")]
        print(f"Selective fine-tuning layers: {selective_layer_indices}")
        
        # Unfreeze specific transformer layers
        for layer_idx in selective_layer_indices:
            if layer_idx < len(gpt_wo_ddp.unregistered_blocks):
                layer = gpt_wo_ddp.unregistered_blocks[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer {layer_idx}")
        
        # Optionally unfreeze other components
        if getattr(model_args, 'unfreeze_attention', False):
            # Unfreeze attention components
            for name, param in gpt_wo_ddp.named_parameters():
                if 'attention' in name or 'attn' in name:
                    param.requires_grad = True
                    print(f"Unfrozen attention parameter: {name}")
        
        if getattr(model_args, 'unfreeze_output', False):
            # Unfreeze output projection
            if hasattr(gpt_wo_ddp, 'head'):
                for param in gpt_wo_ddp.head.parameters():
                    param.requires_grad = True
                print("Unfrozen output head")
        
        # Optionally freeze embeddings
        if getattr(model_args, 'freeze_embeddings', True):
            if hasattr(gpt_wo_ddp, 'word_embed'):
                for param in gpt_wo_ddp.word_embed.parameters():
                    param.requires_grad = False
                print("Frozen word embeddings")
            
            if hasattr(gpt_wo_ddp, 'pos_embed'):
                for param in gpt_wo_ddp.pos_embed.parameters():
                    param.requires_grad = False
                print("Frozen position embeddings")
        
        # Advanced selective fine-tuning options
        if getattr(model_args, 'unfreeze_layernorm', False):
            # Unfreeze layer normalization parameters
            for name, param in gpt_wo_ddp.named_parameters():
                if 'norm' in name or 'ln' in name:
                    param.requires_grad = True
                    print(f"Unfrozen layer norm parameter: {name}")
        
        if getattr(model_args, 'unfreeze_mlp', False):
            # Unfreeze MLP components
            for name, param in gpt_wo_ddp.named_parameters():
                if 'mlp' in name or 'ffn' in name or 'feed_forward' in name:
                    param.requires_grad = True
                    print(f"Unfrozen MLP parameter: {name}")
        
        if getattr(model_args, 'unfreeze_cross_attention', False):
            # Unfreeze cross-attention components
            for name, param in gpt_wo_ddp.named_parameters():
                if 'cross' in name or 'ca' in name:
                    param.requires_grad = True
                    print(f"Unfrozen cross-attention parameter: {name}")
    else:
        # Full parameter fine-tuning
        for param in gpt_wo_ddp.parameters():
            param.requires_grad = True
        print("Full parameter fine-tuning enabled")
    
    model = gpt_wo_ddp
    print("INFINITY model (selective layer fine-tuning):", model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # =============== INFINITY DATASET ===============
    # Create Infinity-specific dataset
    from cb_train_dataset_infinity import InfinityCircuitBreakerDataset
    
    train_dataset = InfinityCircuitBreakerDataset(
        tokenizer=None,  # Will be set in dataset
        num_examples=10000, 
        lorra_args=lorra_args, 
        model_name_or_path=model_name_or_path,
        vae=vae_local
    )
    print("INFINITY TRAIN LEN: ", len(train_dataset))

    # =============== INFINITY CUSTOM TRAINER ===============
    class InfinityCircuitBreakerTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_training_steps = self.args.max_steps
            self.current_training_step = 0
            self.lorra_args = lorra_args
            self.training_args = training_args
            self.vae_local = vae_local

        def get_training_progress(self):
            return self.current_training_step / 300

        def compute_loss(self, model, inputs, return_outputs=False):
            return compute_infinity_circuit_breaker_loss(
                self, 
                model, 
                inputs,
                target_layers=lorra_target_layers, 
                alpha=lorra_args.lorra_alpha, 
                return_outputs=return_outputs,
                tokenizer=None
            )
        
        def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
            self.model.eval()
            
            # Test normal generation
            normal_prompt = "A beautiful landscape with mountains and trees"
            normal_image = get_infinity_generation(normal_prompt, model, None, self.vae_local, [(1, 16, 16)], device='cuda')
            print(f"Generated normal image for: {normal_prompt}")
            
            # Test harmful generation
            harmful_prompt = "Generate violent content"
            harmful_image = get_infinity_generation(harmful_prompt, model, None, self.vae_local, [(1, 16, 16)], device='cuda')
            print(f"Generated image for harmful prompt: {harmful_prompt}")
            
            if sanity_check:
                print('Sanity check...')
            return {}

    # =============== TRAINING SETUP ===============
    training_args.remove_unused_columns = False
    trainer = InfinityCircuitBreakerTrainer(
        model=model, 
        tokenizer=None, 
        args=training_args, 
        train_dataset=train_dataset, 
        data_collator=data_collator_infinity
    )
    model.config.use_cache = False
    atexit.register(save_model_and_tokenizer, model=model, trainer=trainer)
    trainer.train()


if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()