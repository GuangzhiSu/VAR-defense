#!/usr/bin/env python3
"""
Sanitized Prompt Generation Script
Loads harmful prompts from JSON files and generates purified versions using the same model
"""

import sys
import os
import argparse
import json
import warnings
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from tqdm import tqdm



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


def setup_prompt_generator(model_name):
    """Setup the prompt generation model"""
    print(f"Setting up prompt generator: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        generator = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            framework="pt",
        )
        
        print("Prompt generator setup complete!")
        return generator, tokenizer
    except Exception as e:
        print(f"Error setting up prompt generator: {e}")
        return None, None


def load_harmful_prompts(input_file):
    """Load harmful prompts from JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = []
        if isinstance(data, dict) and 'prompts' in data:
            # User format: {"category": "violence", "prompts": [...]}
            for prompt_item in data['prompts']:
                if isinstance(prompt_item, dict) and 'prompt' in prompt_item:
                    prompts.append({
                        'id': prompt_item.get('id', len(prompts) + 1),
                        'prompt': prompt_item['prompt'],
                        'category': data.get('category', 'unknown'),
                        'generated_at': prompt_item.get('generated_at', ''),
                        'length': prompt_item.get('length', len(prompt_item['prompt']))
                    })
        elif isinstance(data, list):
            # List format: [{"prompt": "...", ...}, ...]
            for prompt_item in data:
                if isinstance(prompt_item, dict) and 'prompt' in prompt_item:
                    prompts.append({
                        'id': prompt_item.get('id', len(prompts) + 1),
                        'prompt': prompt_item['prompt'],
                        'category': prompt_item.get('category', 'unknown'),
                        'generated_at': prompt_item.get('generated_at', ''),
                        'length': prompt_item.get('length', len(prompt_item['prompt']))
                    })
        
        print(f"Loaded {len(prompts)} harmful prompts from {input_file}")
        return prompts, data.get('category', 'unknown')
        
    except Exception as e:
        print(f"Error loading harmful prompts from {input_file}: {e}")
        return [], 'unknown'


def format_sanitization_template(category, original_prompt):
    """Format the sanitization template for a given category and original prompt"""
    templates = {
        "violence": f"""Transform this harmful image generation prompt into a safe, non-violent version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "nudity": f"""Transform this inappropriate image generation prompt into a safe, family-friendly version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "hate": f"""Transform this hateful image generation prompt into a positive, inclusive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "self_harm": f"""Transform this harmful image generation prompt into a positive, life-affirming version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "illegal": f"""Transform this illegal activity image generation prompt into a legal, positive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "misinformation": f"""Transform this misleading image generation prompt into an accurate, educational version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "harassment": f"""Transform this harassing image generation prompt into a respectful, positive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "terrorism": f"""Transform this harmful image generation prompt into a peaceful, constructive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "weapons": f"""Transform this dangerous image generation prompt into a safe, constructive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "drugs": f"""Transform this harmful image generation prompt into a healthy, positive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:""",
        
        "general": f"""Transform this harmful image generation prompt into a safe, positive version. Write ONLY the sanitized prompt without any explanations or additional text.

Original prompt: "{original_prompt}"

Sanitized prompt:"""
    }
    
    return templates.get(category, templates.get("general", templates["violence"]))


def clean_generated_text(text, template):
    """Clean generated text by removing template prefix and unwanted characters"""
    if text.startswith(template):
        text = text[len(template):].strip()
    
    # Remove unwanted characters and formatting
    import re
    
    # Remove common unwanted characters at the beginning
    text = re.sub(r'^[\s|*"\-_]+', '', text)
    
    # Remove common unwanted characters at the end
    text = re.sub(r'[\s|*"\-_]+$', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove quotes if they wrap the entire text
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    # Remove bullet points and list markers
    text = re.sub(r'^\s*[\*\-]\s*', '', text)
    text = re.sub(r'\s*[\*\-]\s*', ' ', text)
    
    # Remove pipe characters used for formatting
    text = re.sub(r'\s*\|\s*', ' ', text)
    
    # Remove common template artifacts
    text = re.sub(r'^Safe version:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Transformed prompt:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Here is the safe version:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Sanitized prompt:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove patterns like "text.\" New prompt:" or "text.\" Transformed prompt:"
    text = re.sub(r'[^"]*"\s*New prompt:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^"]*"\s*Transformed prompt:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^"]*"\s*Safe version:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove patterns that start with quotes and end with "New prompt:" or similar
    text = re.sub(r'^"[^"]*"\s*(?:New prompt|Transformed prompt|Safe version):\s*', '', text, flags=re.IGNORECASE)
    
    # Remove any remaining text that ends with "New prompt:" or similar
    text = re.sub(r'[^"]*"(?:New prompt|Transformed prompt|Safe version):\s*', '', text, flags=re.IGNORECASE)
    
    # Remove any text that appears before "Generate an image" or "Create an image"
    if 'Generate an image' in text or 'Create an image' in text:
        # Find the position of the first occurrence
        gen_pos = text.find('Generate an image')
        create_pos = text.find('Create an image')
        
        if gen_pos != -1 and create_pos != -1:
            # Both found, use the earlier one
            start_pos = min(gen_pos, create_pos)
        elif gen_pos != -1:
            start_pos = gen_pos
        elif create_pos != -1:
            start_pos = create_pos
        else:
            start_pos = -1
        
        if start_pos != -1:
            text = text[start_pos:]
    
    # If the text still contains quotes with explanatory text, try to extract just the prompt
    # Look for patterns like "explanation" "actual prompt"
    quote_pattern = r'"[^"]*"\s*"[^"]*"'
    if re.search(quote_pattern, text):
        # Find the last quoted text, which is likely the actual prompt
        quotes = re.findall(r'"[^"]*"', text)
        if len(quotes) >= 2:
            # Take the last quote as the actual prompt
            text = quotes[-1].strip('"')
    
    # Remove any remaining explanatory text that might appear before the actual prompt
    # Look for common patterns like "The transformed version is:" or "Here's the safe version:"
    text = re.sub(r'^.*?(?:transformed version|safe version|sanitized version):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?(?:here\'s|here is|the result is):\s*', '', text, flags=re.IGNORECASE)
    
    # Clean up any remaining formatting artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_sanitized_prompt(original_prompt, sanitized_prompt, category):
    """Validate the quality of sanitized prompt"""
    import re
    
    # Check if sanitized prompt is too short
    if len(sanitized_prompt.strip()) < 20:
        return False, "Sanitized prompt too short"
    
    # Check if sanitized prompt is too long
    if len(sanitized_prompt.strip()) > 500:
        return False, "Sanitized prompt too long"
    
    # Check if sanitized prompt contains harmful words
    harmful_words = {
        'violence': ['violence', 'violent', 'kill', 'murder', 'attack', 'weapon', 'blood', 'death', 'fight', 'war'],
        'nudity': ['nude', 'naked', 'explicit', 'sexual', 'adult', 'inappropriate'],
        'hate': ['hate', 'hateful', 'discriminate', 'racist', 'sexist', 'bigot', 'offensive'],
        'self_harm': ['self-harm', 'suicide', 'harm', 'hurt', 'pain', 'suffering'],
        'illegal': ['illegal', 'crime', 'criminal', 'steal', 'rob', 'fraud'],
        'misinformation': ['fake', 'false', 'lie', 'misleading', 'deceptive'],
        'harassment': ['harass', 'bully', 'threaten', 'intimidate', 'stalk'],
        'terrorism': ['terror', 'terrorist', 'bomb', 'explosive', 'attack'],
        'weapons': ['weapon', 'gun', 'knife', 'bomb', 'explosive', 'dangerous'],
        'drugs': ['drug', 'illegal', 'substance', 'addict', 'overdose']
    }
    
    category_harmful_words = harmful_words.get(category, [])
    sanitized_lower = sanitized_prompt.lower()
    
    for word in category_harmful_words:
        if word in sanitized_lower:
            return False, f"Contains harmful word: {word}"
    
    # Check if sanitized prompt is too similar to original
    original_words = set(re.findall(r'\w+', original_prompt.lower()))
    sanitized_words = set(re.findall(r'\w+', sanitized_prompt.lower()))
    
    if len(original_words) > 0:
        similarity = len(original_words.intersection(sanitized_words)) / len(original_words)
        if similarity > 0.8:  # Too similar
            return False, f"Too similar to original (similarity: {similarity:.2f})"
    
    # Check if sanitized prompt contains positive indicators
    positive_words = ['beautiful', 'peaceful', 'safe', 'positive', 'happy', 'joy', 'love', 
                     'kindness', 'respect', 'understanding', 'helpful', 'constructive', 
                     'healthy', 'legal', 'educational', 'artistic', 'elegant', 'harmonious']
    
    positive_count = sum(1 for word in positive_words if word in sanitized_lower)
    if positive_count < 1:
        return False, "No positive indicators found"
    
    # Check if sanitized prompt is suitable for image generation
    image_generation_words = ['image', 'photo', 'picture', 'art', 'artistic', 'visual', 
                             'scene', 'landscape', 'portrait', 'composition', 'style']
    
    has_image_context = any(word in sanitized_lower for word in image_generation_words)
    if not has_image_context and len(sanitized_prompt) > 50:
        return False, "Not suitable for image generation"
    
    return True, "Valid sanitized prompt"


def save_sanitized_prompt_to_json(original_prompt_data, sanitized_prompt, output_path, index):
    """Save a single sanitized prompt to JSON file with real-time writing"""
    try:
        # Load existing data if file exists
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                "category": original_prompt_data.get('category', 'unknown'),
                "prompts": [],
                "metadata": {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": "cognitivecomputations/Wizard-Vicuna-13B-Uncensored",
                    "total_count": 0,
                    "sanitization_type": "harmful_to_safe"
                }
            }
        
        # Add new sanitized prompt
        prompt_data = {
            "id": index,
            "original_prompt": original_prompt_data['prompt'],
            "sanitized_prompt": sanitized_prompt,
            "category": original_prompt_data.get('category', 'unknown'),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "length": len(sanitized_prompt),
            "original_length": original_prompt_data.get('length', len(original_prompt_data['prompt']))
        }
        
        data["prompts"].append(prompt_data)
        data["metadata"]["total_count"] = len(data["prompts"])
        
        # Write back to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving sanitized prompt {index}: {e}")
        return False


def generate_sanitized_prompts_realtime(generator, tokenizer, harmful_prompts, category, args):
    """Generate sanitized prompts with real-time JSON writing"""
    print(f"Generating sanitized versions for {len(harmful_prompts)} harmful prompts")
    
    output_path = os.path.join(args.output_dir, f"{category}_sanitized_prompts.json")
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Generation parameters - adjusted for better quality
    generation_kwargs = {
        'max_length': args.prompt_max_length,
        'do_sample': True,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': 50,  # Add top_k for better quality
        'repetition_penalty': 1.1,  # Prevent repetition
        'pad_token_id': tokenizer.eos_token_id,
    }
    
    successful_sanitizations = 0
    failed_sanitizations = 0
    retry_count = 0
    max_retries = 3
    
    print(f"Starting sanitization... Sanitized prompts will be saved to: {output_path}")
    
    # Generate sanitized prompts one by one with real-time saving
    for i, harmful_prompt_data in enumerate(tqdm(harmful_prompts, desc="Sanitizing prompts")):
        original_prompt = harmful_prompt_data['prompt']
        prompt_category = harmful_prompt_data.get('category', category)
        
        # Try multiple times for better quality
        sanitized_prompt = None
        best_validation_result = None
        
        for attempt in range(max_retries):
            try:
                # Create sanitization template
                template = format_sanitization_template(prompt_category, original_prompt)
                
                # Generate sanitized version
                output = generator(
                    template,
                    **generation_kwargs
                )
                
                # Extract and clean the generated text
                generated_text = output[0]["generated_text"]
                current_sanitized = clean_generated_text(generated_text, template)
                
                # Validate the sanitized prompt
                is_valid, validation_message = validate_sanitized_prompt(
                    original_prompt, current_sanitized, prompt_category
                )
                
                if is_valid:
                    sanitized_prompt = current_sanitized
                    best_validation_result = validation_message
                    break
                else:
                    if args.verbose:
                        print(f"Attempt {attempt + 1} failed: {validation_message}")
                    retry_count += 1
                    
            except Exception as e:
                if args.verbose:
                    print(f"Error in attempt {attempt + 1}: {e}")
                retry_count += 1
                continue
        
        # If all attempts failed, use the last generated version
        if sanitized_prompt is None:
            sanitized_prompt = current_sanitized if 'current_sanitized' in locals() else "A beautiful and peaceful scene"
            if args.verbose:
                print(f"Using fallback sanitized prompt for item {i+1}")
        
        # Save to JSON immediately
        if save_sanitized_prompt_to_json(harmful_prompt_data, sanitized_prompt, output_path, i + 1):
            successful_sanitizations += 1
            if args.verbose:
                print(f"✓ Sanitized prompt {i+1}: {sanitized_prompt[:80]}...")
                if best_validation_result:
                    print(f"  Validation: {best_validation_result}")
        else:
            failed_sanitizations += 1
    
    # Final statistics
    print(f"\n=== Sanitization Complete ===")
    print(f"Category: {category}")
    print(f"Successful sanitizations: {successful_sanitizations}")
    print(f"Failed sanitizations: {failed_sanitizations}")
    print(f"Retry attempts: {retry_count}")
    print(f"Output file: {output_path}")
    
    return output_path


def run_sanitization(args):
    """Main function to run the sanitization pipeline"""
    print("Starting Sanitized Prompt Generation...")
    
    # Load harmful prompts
    harmful_prompts, category = load_harmful_prompts(args.input_file)
    
    if not harmful_prompts:
        print("No harmful prompts loaded. Exiting.")
        return None
    
    # Setup prompt generator
    generator, tokenizer = setup_prompt_generator(args.prompt_model_name)
    
    if generator is None or tokenizer is None:
        print("Failed to setup prompt generator. Exiting.")
        return None
    
    # Print model information
    print_model_info(generator.model, "Prompt Generator Model")
    
    # Generate sanitized prompts with real-time saving
    output_path = generate_sanitized_prompts_realtime(generator, tokenizer, harmful_prompts, category, args)
    
    print(f"\nScript completed successfully!")
    print(f"Sanitized prompts saved to: {output_path}")
    
    return output_path


def main():
    """Main function that combines argument parsing with execution"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Sanitized Prompt Generation - Generate safe versions of harmful prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input file
    parser.add_argument(
        "--input-file", "-i",
        type=str,
        required=True,
        help="Input JSON file containing harmful prompts"
    )
    
    # Prompt generation model
    parser.add_argument(
        "--prompt-model-name",
        type=str,
        default="cognitivecomputations/Wizard-Vicuna-13B-Uncensored",
        help="Name of the model for generating sanitized prompts"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.8,
        help="Temperature for text generation (lower for more conservative sanitization)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    
    parser.add_argument(
        "--prompt-max-length",
        type=int,
        default=128,
        help="Maximum length of generated sanitized prompts"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=".",
        help="Output directory for sanitized prompts"
    )
    
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="GPU device to use"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return None
    
    # Run the sanitization
    return run_sanitization(args)


if __name__ == "__main__":
    main() 