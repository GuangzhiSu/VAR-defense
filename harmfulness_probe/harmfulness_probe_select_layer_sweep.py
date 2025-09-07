#!/usr/bin/env python3
"""
Harmfulness Probe Select Layer Sweep Experiment
自动遍历select_layer从25到40，记录每层的TPR和FPR，并可视化
"""
import os
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from harmfulness_probe import (
    setup_config, load_models, load_subset, load_llm_model, create_sanitized_prompts,
    collect_training_data, train_detection_model, save_model, load_saved_model,
    test_on_harmful_dataset, test_on_normal_dataset, setup_hook
)

import torch
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_infinity import *

def sweep_select_layer(start_layer=25, end_layer=40, subset_size=100):
    """
    对select_layer从start_layer到end_layer进行遍历，记录每层的TPR和FPR
    """
    # 配置参数
    args, probe_type, _, threshold, max_length = setup_config()
    model, vae, text_tokenizer, text_encoder, tokenizer, device = load_models(args)
    
    # 构建scale_schedule，与harmfulness_probe.py保持一致
    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # 采样较小的数据集
    harmful_subset_path = "/home/gs285/VAR/my_model/new_dataset/harmful_subset.json"
    sanitized_subset_path = "/home/gs285/VAR/my_model/new_dataset/sanitized_prompts.json"
    if os.path.exists(harmful_subset_path):
        harmful_prompts = load_subset(harmful_subset_path)[:subset_size]
    else:
        raise FileNotFoundError(f"{harmful_subset_path} not found.")
    if os.path.exists(sanitized_subset_path):
        sanitized_prompts = load_subset(sanitized_subset_path)[:subset_size]
    else:
        raise FileNotFoundError(f"{sanitized_subset_path} not found.")
    train_data = sanitized_prompts + harmful_prompts

    # 采样测试集
    harmful_test_path = "/home/gs285/VAR/my_model/new_dataset/I2P_sexual_931.csv"
    df_test = pd.read_csv(harmful_test_path)
    harmful_test_prompts = df_test["perturbed_text"].tolist()
    harmful_test_prompts = random.sample(harmful_test_prompts, min(subset_size, len(harmful_test_prompts)))

    # normal测试集
    from datasets import load_dataset
    nm_dataset = load_dataset("sentence-transformers/coco-captions", split="train")
    test_normal = []
    for example in nm_dataset:
        if len(test_normal) < subset_size:
            test_normal.append(example["caption1"])
        else:
            break

    tpr_list = []
    fpr_list = []
    layers = list(range(start_layer, end_layer+1))

    for select_layer in layers:
        print(f"\n===== Running for select_layer={select_layer} =====")
        args.select_layer = select_layer
        # Setup hook
        collected_hidden, hook_handle = setup_hook(model, select_layer)
        # Collect training data
        x_train, y_train = collect_training_data(
            model, vae, text_tokenizer, text_encoder, train_data, collected_hidden, args, harmful_prompts
        )
        hook_handle.remove()
        # Train detection model
        clf = train_detection_model(x_train, y_train, probe_type)

        # Test on harmful prompts (TPR)
        cors = []
        collected_hidden, hook_handle = setup_hook(model, select_layer)
        for prompt in tqdm(harmful_test_prompts, desc=f"Test harmful L{select_layer}"):
            collected_hidden.clear()
            _ = gen_one_img(
                model, vae, text_tokenizer, text_encoder, prompt,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=[3] * len(scale_schedule), tau_list=[1.0] * len(scale_schedule), scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=False
            )
            if len(collected_hidden) == 0:
                continue
            last_hidden = collected_hidden[-1]
            probs = clf.predict_proba(last_hidden)[:, 1]
            max_detection_score = float(np.max(probs))
            cors.append(max_detection_score >= threshold)
        hook_handle.remove()
        tpr = np.mean(cors)
        print(f"Layer {select_layer} TPR: {tpr}")
        tpr_list.append(tpr)

        # Test on normal prompts (FPR)
        cors_normal = []
        collected_hidden, hook_handle = setup_hook(model, select_layer)
        for prompt in tqdm(test_normal, desc=f"Test normal L{select_layer}"):
            collected_hidden.clear()
            _ = gen_one_img(
                model, vae, text_tokenizer, text_encoder, prompt,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=[3] * len(scale_schedule), tau_list=[1.0] * len(scale_schedule), scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=False
            )
            if len(collected_hidden) == 0:
                continue
            last_hidden = collected_hidden[-1]
            probs = clf.predict_proba(last_hidden)[:, 1]
            max_detection_score = float(np.max(probs))
            cors_normal.append(max_detection_score >= threshold)
        hook_handle.remove()
        fpr = np.mean(cors_normal)
        print(f"Layer {select_layer} FPR: {fpr}")
        fpr_list.append(fpr)

    # 可视化
    plt.figure(figsize=(10,6))
    plt.plot(layers, tpr_list, marker='o', label='True Positive Rate (TPR)')
    plt.plot(layers, fpr_list, marker='x', label='False Positive Rate (FPR)')
    plt.xlabel('select_layer')
    plt.ylabel('Rate')
    plt.title('TPR & FPR vs. select_layer')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/gs285/VAR/my_model/harmfulness_probe/select_layer_sweep_results.png')
    plt.show()
    # 保存结果
    with open('select_layer_sweep_results.json', 'w') as f:
        json.dump({'layers': layers, 'tpr': tpr_list, 'fpr': fpr_list}, f, indent=2)

if __name__ == "__main__":
    sweep_select_layer() 