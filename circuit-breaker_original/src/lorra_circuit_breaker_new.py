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
from transformers import Trainer, deepspeed, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# from cb_train_dataset import (
#     CircuitBreakerDataset
# )

# from train_datasets_original import (
#     CircuitBreakerDataset
# )

from cb_train_dataset import (
    CircuitBreakerDataset
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





def compute_infinity_circuit_breaker_loss(self, model, inputs, target_layers, alpha, return_outputs=False, tokenizer=None, **kwargs):
    """
    INFINITY T2I Circuit Breaker Loss Function
    Key differences from LLM version:
    1. Input format: image_tokens + text_cond_tuple instead of text tokens
    2. Model architecture: VAE + Transformer instead of pure LLM
    3. Loss computation: image token space instead of text space
    4. Generation process: autoregressive image token generation + VAE decoding
    """
    self.current_training_step += 1
    log_now = self.current_training_step % 10 == 0

    # === INFINITY INPUT FORMAT ===
    # Extract Infinity-specific inputs
    text_cond_tuple = inputs.get("text_cond_tuple")  # (kv_compact, lens, cu_seqlens_k, max_seqlen_k)
    image_tokens = inputs.get("image_tokens")  # VAE encoded image tokens
    scale_schedule = inputs.get("scale_schedule")  # Dynamic resolution schedule
    
    # Circuit breaker specific inputs
    cb_text_cond_tuple = inputs.get("cb_text_cond_tuple")
    cb_image_tokens = inputs.get("cb_image_tokens")
    cb_scale_schedule = inputs.get("cb_scale_schedule")
    
    # Validation inputs
    val_text_cond_tuple = inputs.get("val_text_cond_tuple")
    val_image_tokens = inputs.get("val_image_tokens")
    val_scale_schedule = inputs.get("val_scale_schedule")

    # === Step Coeff ===
    progress = self.get_training_progress()
    scheduled_coeff = progress
    print(f'\nPROGRESS: {progress:.4f}', '='*50)
    retain_coeff, circuit_breaker_coeff = alpha * scheduled_coeff, alpha * (1-scheduled_coeff)
    
    print(f"retain_coeff: {retain_coeff:.4f} || circuit_breaker_coeff: {circuit_breaker_coeff:.4f}")
    
    # === INFINITY LOSS COMPONENTS ===
    retain_loss = torch.tensor(0.0, device=model.device)
    circuit_breaker_loss = torch.tensor(0.0, device=model.device)
    
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            ### Retain control - normal image generation
            if retain_coeff > 0:
                # INFINITY FORWARD: Use Infinity model format
                orig_retain_outputs = model(
                    label_B_or_BLT=text_cond_tuple,
                    x_BLC_wo_prefix=image_tokens,
                    scale_schedule=scale_schedule
                )
                # Store original hidden states for comparison
                orig_retain_hidden = []
                for layer_idx in target_layers:
                    if hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                        # Extract hidden states from specific layers
                        orig_retain_hidden.append(model.unregistered_blocks[layer_idx].get_hidden_states().detach())
                
                del orig_retain_outputs
                gc.collect()

            ### Circuit Breaker control - harmful prompt handling
            if circuit_breaker_coeff > 0:
                # INFINITY FORWARD: Use Infinity model format
                cb_outputs = model(
                    label_B_or_BLT=cb_text_cond_tuple,
                    x_BLC_wo_prefix=cb_image_tokens,
                    scale_schedule=cb_scale_schedule
                )
                # Store circuit breaker hidden states
                cb_hidden = []
                for layer_idx in target_layers:
                    if hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                        cb_hidden.append(model.unregistered_blocks[layer_idx].get_hidden_states().detach())
                
                del cb_outputs
                gc.collect()

    model.train()

    ### Retain control - ensure normal generation capability
    if retain_coeff > 0:
        # INFINITY FORWARD: Use Infinity model format
        lora_retain_outputs = model(
            label_B_or_BLT=text_cond_tuple,
            x_BLC_wo_prefix=image_tokens,
            scale_schedule=scale_schedule
        )
        
        # Compare hidden states between original and LoRA model
        lora_retain_hidden = []
        for layer_idx in target_layers:
            if hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                lora_retain_hidden.append(model.unregistered_blocks[layer_idx].get_hidden_states())
        
        # Compute retain loss (should be similar to original)
        retain_loss = torch.tensor(0.0, device=model.device)
        for orig_h, lora_h in zip(orig_retain_hidden, lora_retain_hidden):
            retain_loss += F.mse_loss(lora_h, orig_h)
        
        if log_now:
            print(f"\nretain_loss: {retain_loss.item():.4f}")

    ### Circuit Breaker control - prevent harmful generation
    if circuit_breaker_coeff > 0:
        # INFINITY FORWARD: Use Infinity model format
        lora_cb_outputs = model(
            label_B_or_BLT=cb_text_cond_tuple,
            x_BLC_wo_prefix=cb_image_tokens,
            scale_schedule=cb_scale_schedule
        )
        
        # Get LoRA circuit breaker hidden states
        lora_cb_hidden = []
        for layer_idx in target_layers:
            if hasattr(model, 'unregistered_blocks') and layer_idx < len(model.unregistered_blocks):
                lora_cb_hidden.append(model.unregistered_blocks[layer_idx].get_hidden_states())
        
        # Compute circuit breaker loss (should be different from original harmful generation)
        circuit_breaker_loss = torch.tensor(0.0, device=model.device)
        for orig_h, lora_h in zip(cb_hidden, lora_cb_hidden):
            # Use cosine similarity to measure difference
            # We want the LoRA model to produce different representations for harmful prompts
            cos_sim = cosine_similarity(lora_h.flatten(1), orig_h.flatten(1), dim=1)
            circuit_breaker_loss += torch.relu(cos_sim).mean()  # Penalize high similarity
        
        if log_now:
            print(f"circuit_breaker_loss: {circuit_breaker_loss.item():.4f}")
    
    # Combine losses
    total_loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

    print(f"\nretain_loss: {retain_loss.item():.4f} \ncircuit_breaker_loss: {circuit_breaker_loss.item():.4f}")
    print('='*50)

    return (total_loss, ) if return_outputs else total_loss


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
    INFINITY T2I Data Collator
    Key differences from LLM version:
    1. Handles text_cond_tuple format instead of text tokens
    2. Processes image_tokens instead of input_ids
    3. Manages scale_schedule for dynamic resolution
    """
    batch_inputs = {}
    
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k, []).append(input)
    
    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.cat(inputs, dim=0)
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        elif isinstance(inputs[0], tuple):
            # Handle text_cond_tuple format
            batch_inputs[k] = inputs  # Keep as list of tuples
        else:
            raise ValueError(f"Return data type not implemented {type(inputs[0])}")
    
    return batch_inputs

def train():
    """
    Main training function with support for both LLM and Infinity T2I models
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

    device_map = "auto"
    if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
        logging.warning(
            "FSDP and ZeRO3 are both currently incompatible with QLoRA."
        )

    model_name_or_path = model_args.model_name_or_path
    target_layers = lorra_args.target_layers
    transform_layers = lorra_args.transform_layers
    full_layers = lorra_args.full_layers
    
    # =============== INFINITY T2I MODEL SUPPORT ===============
    # Check if we're training an Infinity model
    is_infinity_model = hasattr(model_args, 'model_type') and 'infinity' in model_args.model_type.lower()
    
    if is_infinity_model and INFINITY_AVAILABLE:
        print("="*60)
        print("INFINITY T2I MODEL DETECTED - Using T2I Circuit Breaker")
        print("="*60)

    model_name_or_path = model_args.model_name_or_path
    target_layers = lorra_args.target_layers
    transform_layers = lorra_args.transform_layers
    full_layers = lorra_args.full_layers

    lorra_target_layers = [int(layer) for layer in target_layers.split(",")]
    if "-1" in transform_layers:
        lora_layers_to_transform = [i for i in range(max(lorra_target_layers) + 1)]
    else:
        lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]

    # =============== INFINITY MODEL LOADING ===============
    # Load Infinity model components
    from infinity.utils.load import build_vae_gpt
    from infinity.utils import arg_util
    
    # Create args for Infinity model
    args = arg_util.Args()
    args.model = "infinity_2b"  # or other model size
    args.vae_ckpt = getattr(model_args, 'vae_ckpt', "path/to/vae/checkpoint")
    args.device = "cuda"
    
    # Build VAE and GPT models
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu') if os.path.exists(args.vae_ckpt) else {}
    vae_local, gpt_wo_ddp, gpt_wo_ddp_ema = build_vae_gpt(args, vae_ckpt, skip_gpt=False, device=args.device)
    
    # Load pretrained weights if specified
    if model_name_or_path:
        checkpoint = torch.load(model_name_or_path, map_location='cpu')
        gpt_wo_ddp.load_state_dict(checkpoint, strict=False)
    
    # =============== LoRA CONFIGURATION ===============
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(gpt_wo_ddp, lora_config)
    print("INFINITY model with LoRA:", model)

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

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
    class InfinityCustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_training_steps = self.args.max_steps
            self.current_training_step = 0
            self.lorra_args = lorra_args
            self.training_args = training_args

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
            normal_image = get_infinity_generation(normal_prompt, model, None, vae_local, [(1, 16, 16)], device='cuda')
            print(f"Generated normal image for: {normal_prompt}")
            
            # Test harmful generation
            harmful_prompt = "Generate violent content"
            harmful_image = get_infinity_generation(harmful_prompt, model, None, vae_local, [(1, 16, 16)], device='cuda')
            print(f"Generated image for harmful prompt: {harmful_prompt}")
            
            if sanity_check:
                print('Sanity check...')
            return {}

    # =============== TRAINING SETUP ===============
    training_args.remove_unused_columns = False
    trainer = InfinityCustomTrainer(
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