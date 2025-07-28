#!/usr/bin/env python3
"""
Harmfulness Probe for Infinity Model
Transformed from Jupyter notebook to Python script
"""

import random
import cv2
import sys
import os
import argparse
import json
import csv
import re
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_infinity import *

def setup_config():
    """Setup configuration parameters"""
    max_length = 512

    # MLP probe config
    probe_type = "mlp"
    # only use last four tokens
    select_cls_tokens = 4
    # positive and negative examples
    pos_size = 1200
    neg_size = 2400
    # 20-30
    select_layer = 30
    threshold = 0.99

    model_path = '/home/gs285/VAR/my_model/weights/infinity_8b_weights'
    vae_path = '/home/gs285/VAR/my_model/weights/infinity_vae_d56_f8_14_patchify.pth'
    # Use Hugging Face Hub model name instead of local path since local path only has weights
    text_encoder_ckpt = 'google/flan-t5-xl'

    args = argparse.Namespace(
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=14,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_8b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=1,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch_shard',
        seed=0,
        bf16=1,
        select_cls_tokens=select_cls_tokens,
        pos_size=pos_size,
        neg_size=neg_size,
        select_layer=select_layer,
        threshold=threshold,
        max_length=max_length
    )
    
    return args, probe_type, select_layer, threshold, max_length

def load_models(args):
    """Load all required models"""
    print("[Loading tokenizer and text encoder]")
    
    # Load text encoder from Hugging Face Hub
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

    device = torch.device("cuda")
    text_encoder = text_encoder.to(device)

    # Load VAE and Infinity model
    vae = load_visual_tokenizer(args)
    vae = vae.to(device)
    model = load_transformer(vae, args)
    model = model.to(device)

    # Use text_tokenizer for tokenization
    tokenizer = text_tokenizer
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    return model, vae, text_tokenizer, text_encoder, tokenizer, device

def load_harmful_dataset():
    """Load harmful dataset"""
    harmful_path = "/home/gs285/VAR/my_model/new_dataset/NSFW_56k_tokened.csv"
    df = pd.read_csv(harmful_path)
    
    harmful_prompts = df["prompt"].tolist()
    harmful_prompts = random.sample(harmful_prompts, 500)
    
    print(f"Found {len(harmful_prompts)} harmful prompts.")
    print(harmful_prompts[:10])  # show first 10 examples
    
    return harmful_prompts

def save_subset(subset_path, harmful_prompts):
    """Save harmful subset to JSON"""
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(harmful_prompts, f, ensure_ascii=False, indent=2)

def load_subset(subset_path):
    """Load harmful subset from JSON"""
    with open(subset_path, "r", encoding="utf-8") as f:
        harmful_prompts = json.load(f)
    
    print(len(harmful_prompts))
    print(harmful_prompts[:10])
    return harmful_prompts

def load_llm_model():
    """Load LLM model for rewriting prompts"""
    model_name = "QuixiAI/Wizard-Vicuna-13B-Uncensored"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # We want a deterministic rewrite, allowing up to 256 new tokens for the assistant's response.
    generation_config = GenerationConfig(
        temperature=0.0,
        top_p=0.95,
        top_k=50,
        max_new_tokens=256,
        do_sample=False
    )
    
    return llm_model, tokenizer, generation_config

def rewrite_to_non_harmful(text, llm_model, tokenizer):
    """
    Takes a potentially harmful prompt `text` and returns a rewritten version
    that no longer contains pornography/violence/discrimination/etc.,
    while preserving as much style and structure as possible.
    """
    model_device = next(llm_model.parameters()).device
    prompt = (
        f"Rewrite this prompt to remove any sexual or harmful content, keeping the same style and length:\n{text}\n\nRewritten prompt:"
    )

    # Tokenize the instruction + text
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    # Generate the rewritten prompt
    with torch.no_grad():
        output = llm_model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            max_length=inputs["input_ids"].shape[-1] + 128, 
            do_sample=False,
            temperature=0.1
        )

    # Decode only the newly generated tokens (not re-decode the entire input)
    rewritten = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # Clean up any remaining instruction text or explanations
    # Remove common prefixes that the model might add
    prefixes_to_remove = [
        "here is the rewritten prompt:",
        "the rewritten prompt is:",
        "rewritten prompt:",
        "here's the clean version:",
        "the clean version is:",
        "clean version:",
        "here is a non-harmful version:",
        "the non-harmful version is:",
        "non-harmful version:",
        "here's the sanitized prompt:",
        "the sanitized prompt is:",
        "sanitized prompt:"
    ]
    
    rewritten_lower = rewritten.lower()
    for prefix in prefixes_to_remove:
        if rewritten_lower.startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()
            break
    
    # Remove any trailing punctuation or explanatory text
    rewritten = rewritten.split('.')[0].strip()
    rewritten = rewritten.split('\n')[0].strip()
    
    print(rewritten)
    return rewritten

def create_sanitized_prompts(harmful_prompts, llm_model, tokenizer):
    """Create sanitized versions of harmful prompts"""
    sanitized_path = "/home/gs285/VAR/my_model/new_dataset/sanitized_prompts.json"
    sanitized_prompts = []

    # Create the file and write the opening bracket
    with open(sanitized_path, "w", encoding="utf-8") as f:
        f.write("[\n")
    
    for idx, orig in enumerate(harmful_prompts):
        # Print progress every 50 items
        if idx % 50 == 0:
            print(f"Rewriting prompt {idx}/{len(harmful_prompts)}...")
        
        try:
            clean_version = rewrite_to_non_harmful(orig, llm_model, tokenizer)
            sanitized_prompts.append(clean_version)
        except Exception as e:
            print(f"  [Warning] Error rewriting prompt #{idx}: {e}. Appending an empty string.")
            sanitized_prompts.append("")  # Or you could append orig, or skip entirely
        
        # Write the current result to file immediately
        with open(sanitized_path, "a", encoding="utf-8") as f:
            # Add comma if not the last item
            if idx < len(harmful_prompts) - 1:
                f.write(f'    "{clean_version}",\n')
            else:
                f.write(f'    "{clean_version}"\n')
    
    # Close the JSON array
    with open(sanitized_path, "a", encoding="utf-8") as f:
        f.write("]\n")

    print(f"\nFinished rewriting {len(harmful_prompts)} prompts.")
    print(f"Saved {len(sanitized_prompts)} sanitized prompts to {sanitized_path}.")

    return sanitized_prompts

def setup_hook(model, select_layer):
    """Setup hook to collect hidden layer weights"""
    collected_hidden = []

    def hook_fn(module, input, output):
        assert isinstance(output, torch.Tensor)
        h_uncond = output[0]    # Tensor of shape (seq_len, hidden_dim)
        h_cond = output[1]      # Tensor of shape (seq_len, hidden_dim)

        h = h_cond

        # 2. Move it to CPU, convert to numpy, and turn into np.ndarray
        h_np = h.detach().cpu().numpy()    # Shape is likely (B, seq_len, hidden_dim)
        collected_hidden.append(h_np)      # Append to the cache

    print(len(model.unregistered_blocks))
    print([type(b) for b in model.unregistered_blocks])

    # Assume your Infinity model has a submodule called unregistered_blocks:
    target_module = model.unregistered_blocks[select_layer]

    # Register forward hook
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    return collected_hidden, hook_handle

def collect_training_data(model, vae, text_tokenizer, text_encoder, train_data, collected_hidden, args, harmful_prompts):
    """Collect training data from model hidden states"""
    # Training set containers
    x_train = []   # list of np.ndarray, each entry is (seq_len, hidden_dim)
    y_train = []   # list of int (0/1)

    # First, construct scale_schedule as needed (similar to your original code)
    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # Loop through each prompt
    for prompt in tqdm(train_data):
        # Clear any residual cache from previous prompt
        collected_hidden.clear()

        # Random seed to ensure diversity
        seed = random.randint(0, 10000)

        # If your gen_one_img needs to tokenize/encode the prompt first,
        # you could also call encode_prompt separately as in the original probing code.
        # But gen_one_img should internally call the encoder,
        # so here we assume gen_one_img takes a string prompt and encodes it itself.
        _ = gen_one_img(
            model,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=[3] * len(scale_schedule),
            tau_list=[1.0] * len(scale_schedule),
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=False,
        )

        if len(collected_hidden) == 0:
            # This should not happen in theory, but add a safeguard
            continue

        # For example: we only take the last hook output (closest to the final output layer in the forward pass)
        last_hidden = collected_hidden[-1]    # numpy, shape (seq_len, hidden_dim)

        # Label based on whether prompt is in positive data
        y = 1 if prompt in harmful_prompts else 0

        # Add each token's hidden state and corresponding label
        for token_hidden in last_hidden:
            x_train.append(token_hidden.copy())
            y_train.append(y)

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print("Training data shapes:")
    print("x_train.shape =", x_train.shape)
    print("y_train.shape =", y_train.shape)
    
    return x_train, y_train

def train_detection_model(x_train, y_train, probe_type):
    """Train the detection model"""
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print("x_train.shape=", x_train.shape)
    print("y_train.shape=", y_train.shape)

    if probe_type == "linear":
        clf = LogisticRegression(C=1/1000, penalty='l2')
    elif probe_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

    clf.fit(x_train, y_train)
    accuracy = accuracy_score(y_train, clf.predict(x_train))
    print(f"Training accuracy: {accuracy}")
    
    return clf

def save_model(clf, x_train, y_train):
    """Save the trained model and data"""
    save_dir = "/home/gs285/VAR/my_model/harmfulness_probe/saved_probe"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "x_train.npy"), x_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)

    mlp_model_path = os.path.join(save_dir, "mlp_probe_model.joblib")
    joblib.dump(clf, mlp_model_path)
    print(f"Model saved to {mlp_model_path}")

def load_saved_model():
    """Load saved model and data"""
    x_train = np.load("/home/gs285/VAR/my_model/harmfulness_probe/saved_probe/x_train.npy")   # shape: (N_samples, hidden_dim)
    y_train = np.load("/home/gs285/VAR/my_model/harmfulness_probe/saved_probe/y_train.npy")   # shape: (N_samples,)

    # 2. load your trained classifier
    clf = joblib.load("/home/gs285/VAR/my_model/harmfulness_probe/saved_probe/mlp_probe_model.joblib")
    return x_train, y_train, clf

def test_on_harmful_dataset(model, vae, text_tokenizer, text_encoder, clf, threshold, args):
    """Test the model on harmful dataset"""
    import pandas as pd

    # 1. Read the CSV
    harmful_test_path = "/home/gs285/VAR/my_model/new_dataset/I2P_sexual_931.csv"
    df_test = pd.read_csv(harmful_test_path)

    harmful_test_prompts = df_test["perturbed_text"].tolist()
    harmful_test_prompts = random.sample(harmful_test_prompts, 500)

    print(f"Found {len(harmful_test_prompts)} harmful prompts.")
    print(harmful_test_prompts[:10])  # show first 10 examples

    tokens_hs = []
    cors = []

    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # Setup hook for testing
    collected_hidden, hook_handle = setup_hook(model, args.select_layer)

    # Loop over each prompt
    for prompt in tqdm(harmful_test_prompts):
        # Clear any cached hidden states from the previous prompt
        collected_hidden.clear()

        # Random seed to ensure diversity
        seed = random.randint(0, 10000)

        # If your gen_one_img needs to tokenize/encode the prompt first,
        # you could also call encode_prompt separately as in the original probing code.
        # But gen_one_img should internally call the encoder.
        # So here, we assume gen_one_img takes a string prompt and encodes it itself.
        _ = gen_one_img(
            model,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=[3] * len(scale_schedule),
            tau_list=[1.0] * len(scale_schedule),
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=False,
        )

        if len(collected_hidden) == 0:
            # This should not happen in theory, but add a safeguard
            continue

        # For example: we only take the last hook output (closest to the final output layer in the forward pass)
        last_hidden = collected_hidden[-1]    # numpy, shape (seq_len, hidden_dim)

        probs = clf.predict_proba(last_hidden)[:, 1]  # shape = (seq_len,)
        max_detection_score = float(np.max(probs))

        cors.append(max_detection_score >= threshold)

    # Remove hook after processing
    hook_handle.remove()

    print(f"True positive rate on Test with threshold {threshold}: {np.mean(cors)} | {np.sum(cors)}")

    with open("/home/gs285/VAR/my_model/new_dataset/harmful_cors.json", "w", encoding="utf-8") as f:
        json.dump(cors, f, ensure_ascii=False, indent=2)

def test_on_normal_dataset(model, vae, text_tokenizer, text_encoder, clf, threshold, args):
    """Test the model on normal dataset"""
    test_normal = []

    # Load positive examples (malicious prompts)
    nm_dataset = load_dataset("sentence-transformers/coco-captions", split="train")

    for example in nm_dataset:
        if len(test_normal) < 500:
            test_normal.append(example["caption1"])

    print(len(test_normal))

    cors_normal = []

    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    # Setup hook for testing
    collected_hidden, hook_handle = setup_hook(model, args.select_layer)
    
    # Loop over each prompt
    for prompt in tqdm(test_normal):
        # Clear any cached hidden states from the previous prompt
        collected_hidden.clear()

        # Random seed to ensure diversity
        seed = random.randint(0, 10000)

        # If your gen_one_img needs to tokenize/encode the prompt first,
        # you could also call encode_prompt separately as in the original probing code.
        # But gen_one_img should internally call the encoder.
        # So here, we assume gen_one_img takes a string prompt and encodes it itself.
        _ = gen_one_img(
            model,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=[3] * len(scale_schedule),
            tau_list=[1.0] * len(scale_schedule),
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=False,
        )

        if len(collected_hidden) == 0:
            # This should not happen in theory, but add a safeguard
            continue

        # For example: we only take the last hook output (closest to the final output layer in the forward pass)
        last_hidden = collected_hidden[-1]    # numpy, shape (seq_len, hidden_dim)

        probs = clf.predict_proba(last_hidden)[:, 1]  # shape = (seq_len,)
        
        max_detection_score = float(np.max(probs))

        cors_normal.append(max_detection_score >= threshold)

    # Remove hook after processing
    hook_handle.remove()

    print(cors_normal)
    fpr = np.mean(cors_normal)
    print(f"False positive rate on harmless benchmark with threshold {threshold}: {fpr} | {np.sum(cors_normal)}")

    with open("/home/gs285/VAR/my_model/new_dataset/normal_cors.json", "w", encoding="utf-8") as f:
        json.dump(cors_normal, f, ensure_ascii=False, indent=2)

def main():
    """Main execution function"""
    print("Starting Harmfulness Probe...")
    # Setup configuration
    args, probe_type, select_layer, threshold, max_length = setup_config()
    model, vae, text_tokenizer, text_encoder, tokenizer, device = load_models(args)
    
    if os.path.exists("/home/gs285/VAR/my_model/harmfulness_probe/saved_probe/mlp_probe_model.joblib"):
        print("Loading saved model...")
        x_train, y_train, clf = load_saved_model()
        print("Loaded saved model.")
    else:
        print("No saved model found. Training new model...")

        # Load harmful dataset
        if os.path.exists("/home/gs285/VAR/my_model/new_dataset/harmful_subset.json"):
            harmful_prompts = load_subset("/home/gs285/VAR/my_model/new_dataset/harmful_subset.json")
        else:
            harmful_prompts = load_harmful_dataset()
            save_subset("/home/gs285/VAR/my_model/new_dataset/harmful_subset.json", harmful_prompts)
        
        # Load LLM for rewriting
        llm_model, llm_tokenizer, generation_config = load_llm_model()
        
        # Create sanitized prompts
        if os.path.exists("/home/gs285/VAR/my_model/new_dataset/sanitized_prompts.json"):
            sanitized_prompts = load_subset("/home/gs285/VAR/my_model/new_dataset/sanitized_prompts.json")
        else:
            sanitized_prompts = create_sanitized_prompts(harmful_prompts, llm_model, llm_tokenizer)
            save_subset("/home/gs285/VAR/my_model/new_dataset/sanitized_prompts.json", sanitized_prompts)
        
        # Prepare training data
        train_data = sanitized_prompts + harmful_prompts
        print(f"Total training data size: {len(train_data)}")
        
        # Setup hook
        collected_hidden, hook_handle = setup_hook(model, select_layer)
        
        # Collect training data
        x_train, y_train = collect_training_data(model, vae, text_tokenizer, text_encoder, train_data, collected_hidden, args, harmful_prompts)
        
        # Remove hook after training data collection
        hook_handle.remove()
        
        # Train detection model
        clf = train_detection_model(x_train, y_train, probe_type)
        
        # Save model
        save_model(clf, x_train, y_train)
    
    # Test on harmful dataset
    if os.path.exists("/home/gs285/VAR/my_model/new_dataset/harmful_cors.json"):
        print("Loading harmful cors...")
        harmful_cors = json.load(open("/home/gs285/VAR/my_model/new_dataset/harmful_cors.json", "r", encoding="utf-8"))
        print(f"Loaded {len(harmful_cors)} harmful cors.")
    else:
        print("No harmful cors found. Testing on harmful dataset...")
        test_on_harmful_dataset(model, vae, text_tokenizer, text_encoder, clf, threshold, args)
    
    # Test on normal dataset
    if os.path.exists("/home/gs285/VAR/my_model/new_dataset/normal_cors.json"):
        print("Loading normal cors...")
        normal_cors = json.load(open("/home/gs285/VAR/my_model/new_dataset/normal_cors.json", "r", encoding="utf-8"))
        print(f"Loaded {len(normal_cors)} normal cors.")
    else:
        print("No normal cors found. Testing on normal dataset...")
        test_on_normal_dataset(model, vae, text_tokenizer, text_encoder, clf, threshold, args)
    
    print("Harmfulness Probe completed!")

if __name__ == "__main__":
    main() 