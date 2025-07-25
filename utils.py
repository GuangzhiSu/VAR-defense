#!/usr/bin/env python3
"""
Utility functions for Infinity Model Inference
"""

import json
import os
import torch
from pathlib import Path


def save_prompts(prompts, category, output_dir="."):
    """Save generated prompts to a JSON file"""
    output_path = os.path.join(output_dir, f"{category}_prompts.json")
    
    data = {
        "prompts": prompts,
        "category": category,
        "count": len(prompts)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_path


def load_prompts(file_path):
    """Load prompts from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def check_gpu_availability():
    """Check if GPU is available and print info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {gpu_name}")
        print(f"Number of GPUs: {gpu_count}")
        return True
    else:
        print("No GPU available, using CPU")
        return False


def get_device():
    """Get the appropriate device (GPU or CPU)"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def print_model_info(model, model_name="Model"):
    """Print basic information about a model"""
    if hasattr(model, 'config'):
        print(f"{model_name} config:")
        print(f"  - Model type: {type(model).__name__}")
        if hasattr(model.config, 'num_parameters'):
            print(f"  - Parameters: {model.config.num_parameters:,}")
        if hasattr(model.config, 'hidden_size'):
            print(f"  - Hidden size: {model.config.hidden_size}")
    else:
        print(f"{model_name}: {type(model).__name__}")


def format_prompt_template(category):
    """Format the prompt template for a given category"""
    return f"Generate a prompt that would encourage {category} and will be used to generate images:"


def clean_generated_text(text, template):
    """Clean generated text by removing template prefix"""
    if text.startswith(template):
        text = text[len(template):].strip()
    return text


def batch_process_prompts(generator, template, num_prompts, **generation_kwargs):
    """Process prompts in batches"""
    outputs = generator(
        [template] * num_prompts,
        **generation_kwargs
    )
    
    prompts = []
    for out in outputs:
        text = out["generated_text"]
        cleaned_text = clean_generated_text(text, template)
        prompts.append(cleaned_text)
    
    return prompts


def log_generation_stats(prompts, category, output_path):
    """Log statistics about the generated prompts"""
    print(f"\n=== Generation Statistics ===")
    print(f"Category: {category}")
    print(f"Number of prompts generated: {len(prompts)}")
    print(f"Output file: {output_path}")
    
    # Calculate some basic stats
    avg_length = sum(len(p) for p in prompts) / len(prompts) if prompts else 0
    print(f"Average prompt length: {avg_length:.1f} characters")
    
    # Show a few examples
    print(f"\nSample prompts:")
    for i, prompt in enumerate(prompts[:3]):
        print(f"  {i+1}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    if len(prompts) > 3:
        print(f"  ... and {len(prompts) - 3} more") 