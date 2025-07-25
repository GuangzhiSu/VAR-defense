#!/usr/bin/env python3
"""
Script for generating Infinity model finetuning parameters using conditional generation
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conditional_generate import (
    generate_infinity_finetuning_params, 
    save_finetuning_params,
    InfinityFinetuningParameterGenerator
)


def get_infinity_model_shapes(model_path):
    """Extract parameter shapes from Infinity model checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Extract shapes for key parameters
        target_shapes = {}
        for key, param in state_dict.items():
            # Focus on attention and MLP parameters
            if any(x in key for x in ['attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']):
                target_shapes[key] = param.shape
        
        return target_shapes
    except Exception as e:
        print(f"Warning: Could not extract shapes from {model_path}: {e}")
        # Return default shapes
        return {
            "blocks.0.attn.qkv.weight": (1024, 1024),
            "blocks.0.attn.proj.weight": (1024, 1024),
            "blocks.0.mlp.fc1.weight": (4096, 1024),
            "blocks.0.mlp.fc2.weight": (1024, 4096),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Infinity model finetuning parameters")
    parser.add_argument("--infinity_model_path", type=str, default=None,
                       help="Path to Infinity model checkpoint")
    parser.add_argument("--text_prompt", type=str, 
                       default="Generate finetuning parameters for improved performance",
                       help="Text prompt for conditional generation")
    parser.add_argument("--num_generated", type=int, default=5,
                       help="Number of parameter sets to generate")
    parser.add_argument("--output_dir", type=str, default="./generated_finetuning_params",
                       help="Output directory for generated parameters")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for generation")
    parser.add_argument("--save_config", action="store_true",
                       help="Save generation configuration")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get target shapes from Infinity model
    target_shapes = None
    if args.infinity_model_path and os.path.exists(args.infinity_model_path):
        print(f"Extracting parameter shapes from: {args.infinity_model_path}")
        target_shapes = get_infinity_model_shapes(args.infinity_model_path)
        print(f"Found {len(target_shapes)} parameter shapes")
    else:
        print("Using default parameter shapes")
        target_shapes = {
            "blocks.0.attn.qkv.weight": (1024, 1024),
            "blocks.0.attn.proj.weight": (1024, 1024),
            "blocks.0.mlp.fc1.weight": (4096, 1024),
            "blocks.0.mlp.fc2.weight": (1024, 4096),
        }
    
    # Save target shapes
    shapes_path = os.path.join(args.output_dir, "target_shapes.json")
    with open(shapes_path, 'w') as f:
        json.dump({k: list(v) for k, v in target_shapes.items()}, f, indent=2)
    print(f"Target shapes saved to: {shapes_path}")
    
    # Generate parameters
    print(f"\nGenerating {args.num_generated} parameter sets...")
    print(f"Text prompt: {args.text_prompt}")
    print(f"Device: {args.device}")
    print("-" * 60)
    
    generated_params = generate_infinity_finetuning_params(
        infinity_model_path=args.infinity_model_path,
        text_prompt=args.text_prompt,
        target_shapes=target_shapes,
        device=args.device,
        num_generated=args.num_generated
    )
    
    # Save parameters
    print("\nSaving generated parameters...")
    for i, params in enumerate(generated_params):
        save_path = os.path.join(args.output_dir, f"finetuning_params_{i+1:03d}.pth")
        save_finetuning_params(params, save_path)
    
    # Save generation config
    if args.save_config:
        config = {
            "infinity_model_path": args.infinity_model_path,
            "text_prompt": args.text_prompt,
            "num_generated": args.num_generated,
            "device": args.device,
            "target_shapes": {k: list(v) for k, v in target_shapes.items()},
            "generated_files": [f"finetuning_params_{i+1:03d}.pth" for i in range(args.num_generated)]
        }
        
        config_path = os.path.join(args.output_dir, "generation_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Generation config saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("Parameter generation completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {args.num_generated} parameter sets")


def generate_with_preset_prompts():
    """Generate parameters with preset prompts for different use cases"""
    
    preset_prompts = {
        "performance": "Generate finetuning parameters for improved performance and accuracy",
        "robustness": "Generate finetuning parameters that enhance model robustness against adversarial attacks",
        "efficiency": "Generate finetuning parameters that optimize computational efficiency",
        "privacy": "Generate finetuning parameters that protect user privacy and prevent data leakage",
        "fairness": "Generate finetuning parameters that ensure fair and unbiased model behavior",
        "security": "Generate finetuning parameters that enhance model security against malicious inputs",
        "adaptability": "Generate finetuning parameters that improve model adaptability to new domains",
        "interpretability": "Generate finetuning parameters that enhance model interpretability and explainability"
    }
    
    print("Available preset prompts:")
    for key, prompt in preset_prompts.items():
        print(f"  {key}: {prompt}")
    
    # Generate for each preset
    for key, prompt in preset_prompts.items():
        print(f"\nGenerating parameters for: {key}")
        output_dir = f"./generated_finetuning_params_{key}"
        
        generated_params = generate_infinity_finetuning_params(
            text_prompt=prompt,
            device="cpu",  # Use CPU for demo
            num_generated=2
        )
        
        # Save parameters
        os.makedirs(output_dir, exist_ok=True)
        for i, params in enumerate(generated_params):
            save_path = os.path.join(output_dir, f"finetuning_params_{i+1:03d}.pth")
            save_finetuning_params(params, save_path)


if __name__ == "__main__":
    # Check if running with preset prompts
    if len(sys.argv) == 1:
        print("No arguments provided. Running with preset prompts...")
        generate_with_preset_prompts()
    else:
        main() 