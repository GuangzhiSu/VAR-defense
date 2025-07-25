#!/usr/bin/env python3
"""
Infinity Model Inference Script
Converts the Jupyter notebook functionality to a standalone Python script
"""

import random
import cv2
import sys
import os
import argparse
import json
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import torch
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer, TextGenerationPipeline
from tqdm import tqdm

# Import from tools.run_infinity
from run_infinity import *

# Import local modules
from utils import (
    save_prompts, check_gpu_availability, get_device, create_output_directory,
    print_model_info, format_prompt_template, batch_process_prompts, log_generation_stats
)


def create_args_from_parsed(args):
    """Create argparse.Namespace object for Infinity model from parsed arguments"""
    return argparse.Namespace(
        pn='1M',
        model_path=args.model_path,
        cfg_insertion_layer=0,
        vae_type=14,
        vae_path=args.vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_8b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=args.text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=1,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch_shard',
        seed=0,
        bf16=1
    )


def validate_paths(args):
    """Validate that all required paths exist"""
    paths_to_check = [
        args.model_path,
        args.vae_path,
        args.text_encoder_ckpt
    ]
    
    missing_paths = []
    for path in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("Warning: The following paths do not exist:")
        for path in missing_paths:
            print(f"  - {path}")
        print("The script will attempt to use fallback options where available.")
    
    return len(missing_paths) == 0


def load_models(args):
    """Load all required models"""
    print("Loading models...")
    
    # Load text tokenizer and encoder
    try:
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    except OSError:
        print("Local path failed, falling back to model hub...")
        text_tokenizer, text_encoder = load_tokenizer(t5_path="google/flan-t5-xl")

    device = get_device()
    text_encoder = text_encoder.to(device)

    # Load VAE and Infinity model
    vae = load_visual_tokenizer(args)
    vae = vae.to(device)
    model = load_transformer(vae, args)
    model = model.to(device)

    # Configure tokenizer
    tokenizer = text_tokenizer
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    print("Models loaded successfully!")
    return text_tokenizer, text_encoder, vae, model, tokenizer, device


def setup_prompt_generator(args):
    """Setup the prompt generation model"""
    print("Setting up prompt generator...")
    
    model_name = args.prompt_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",             # automatically place layers on GPU(s)
        torch_dtype=torch.float16,     # use FP16 for memory savings
        trust_remote_code=True         # in case the repo defines custom modeling code
    )
    
    # Fix the device issue by not specifying device parameter
    generator = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
    )
    
    print("Prompt generator setup complete!")
    return generator, tokenizer


def generate_harmful_prompts(generator, tokenizer, args):
    """Generate harmful prompts for the specified category"""
    print(f"Generating {args.num_prompts} prompts for category: {args.category}")
    
    template = format_prompt_template(args.category)
    
    # Generation parameters
    generation_kwargs = {
        'max_length': args.prompt_max_length,
        'do_sample': True,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'pad_token_id': tokenizer.eos_token_id,
    }
    
    # Generate prompts
    prompts = batch_process_prompts(generator, template, args.num_prompts, **generation_kwargs)
    
    # Save prompts
    output_path = save_prompts(prompts, args.category, args.output_dir)
    
    # Log statistics
    log_generation_stats(prompts, args.category, output_path)
    
    return prompts, output_path


def run_inference(args):
    """Main function to run the inference pipeline"""
    print("Starting Infinity Model Inference...")
    
    # Check GPU availability
    check_gpu_availability()
    
    # Validate paths
    validate_paths(args)
    
    # Create args object for Infinity model
    infinity_args = create_args_from_parsed(args)
    
    # Load models
    text_tokenizer, text_encoder, vae, model, tokenizer, device = load_models(infinity_args)
    
    # Print model information
    print_model_info(model, "Infinity Model")
    print_model_info(text_encoder, "Text Encoder")
    
    # Setup prompt generator
    generator, gen_tokenizer = setup_prompt_generator(args)
    
    # Generate harmful prompts
    prompts, output_path = generate_harmful_prompts(generator, gen_tokenizer, args)
    
    print(f"\nScript completed successfully!")
    print(f"Generated prompts saved to: {output_path}")
    
    return prompts, output_path





def main():
    """Main function that combines argument parsing with execution"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Infinity Model Inference - Generate harmful prompts for testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default='/home/gs285/VAR/Infinity/weights/infinity_8b_weights',
        help="Path to Infinity model weights"
    )
    
    parser.add_argument(
        "--vae-path",
        type=str,
        default='/home/gs285/VAR/Infinity/weights/infinity_vae_d56_f8_14_patchify.pth',
        help="Path to VAE weights"
    )
    
    parser.add_argument(
        "--text-encoder-ckpt",
        type=str,
        default='/home/gs285/VAR/Infinity/weights/flan-t5-xl-official',
        help="Path to text encoder checkpoint"
    )
    
    # Prompt generation model
    parser.add_argument(
        "--prompt-model-name",
        type=str,
        default="cognitivecomputations/Wizard-Vicuna-13B-Uncensored",
        help="Name of the model for generating prompts"
    )
    
    # Generation parameters
    parser.add_argument(
        "--category", "-c",
        type=str,
        default="violence",
        help="Category of harmful content to generate prompts for"
    )
    
    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=200,
        help="Number of prompts to generate"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Temperature for text generation"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum length for Infinity model"
    )
    
    parser.add_argument(
        "--prompt-max-length",
        type=int,
        default=64,
        help="Maximum length of generated prompts"
    )
    
    # MLP probe config
    parser.add_argument(
        "--probe-type",
        type=str,
        default="mlp",
        help="Type of probe to use"
    )
    
    parser.add_argument(
        "--select-cls-tokens",
        type=int,
        default=4,
        help="Number of CLS tokens to select"
    )
    
    parser.add_argument(
        "--pos-size",
        type=int,
        default=1200,
        help="Number of positive examples"
    )
    
    parser.add_argument(
        "--neg-size",
        type=int,
        default=2400,
        help="Number of negative examples"
    )
    
    parser.add_argument(
        "--select-layer",
        type=int,
        default=30,
        help="Layer to select for analysis"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Threshold value"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Output directory for generated prompts"
    )
    
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check GPU if requested
    if args.check_gpu:
        check_gpu_availability()
        return
    
    # Run the inference
    return run_inference(args)


if __name__ == "__main__":
    main() 