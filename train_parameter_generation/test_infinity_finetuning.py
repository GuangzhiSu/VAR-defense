#!/usr/bin/env python3
"""
Test script for Infinity finetuning parameter generation system
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infinity_encoder import InfinityModelEncoder, EnhancedTextEncoder, create_condition_fusion
from conditional_generate import (
    LoRATokenizer, 
    RecurrentPrototypeModule, 
    InfinityFinetuningParameterGenerator,
    generate_infinity_finetuning_params
)


def test_lora_tokenizer():
    """Test LoRA tokenizer"""
    print("Testing LoRA Tokenizer...")
    
    tokenizer = LoRATokenizer(token_size=64)
    
    # Create a dummy LoRA matrix
    lora_matrix = torch.randn(128, 256)
    
    # Tokenize
    tokens = tokenizer.lora_matrix_to_tokens(lora_matrix, layer_idx=0)
    print(f"Original matrix shape: {lora_matrix.shape}")
    print(f"Tokenized shape: {tokens.shape}")
    
    # Convert back
    reconstructed = tokenizer.tokens_to_lora_matrix(tokens, lora_matrix.shape)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check reconstruction error
    error = torch.mean((lora_matrix - reconstructed) ** 2).item()
    print(f"Reconstruction MSE: {error:.6f}")
    
    return error < 1e-3


def test_recurrent_module():
    """Test recurrent prototype module"""
    print("\nTesting Recurrent Prototype Module...")
    
    recurrent = RecurrentPrototypeModule(token_dim=96, hidden_dim=256)
    
    # Create dummy tokens
    batch_size, seq_len = 2, 10
    tokens = torch.randn(batch_size, seq_len, 96)
    
    # Process through recurrent module
    prototypes, (h_n, c_n) = recurrent(tokens)
    
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Output prototypes shape: {prototypes.shape}")
    print(f"Hidden state shape: {h_n.shape}")
    
    return prototypes.shape == (batch_size, seq_len, 256)


def test_encoders():
    """Test encoders"""
    print("\nTesting Encoders...")
    
    # Test base model encoder
    base_encoder = InfinityModelEncoder(embed_dim=512)
    base_embedding = base_encoder()
    print(f"Base embedding shape: {base_embedding.shape}")
    
    # Test text encoder
    text_encoder = EnhancedTextEncoder(embed_dim=512)
    text_embedding = text_encoder("Generate finetuning parameters for improved performance")
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Test fusion
    fusion = create_condition_fusion()
    condition = fusion(torch.cat([base_embedding, text_embedding], dim=-1))
    print(f"Fused condition shape: {condition.shape}")
    
    return (base_embedding.shape[1] == 512 and 
            text_embedding.shape[1] == 512 and 
            condition.shape[1] == 1024)


def test_parameter_generator():
    """Test parameter generator"""
    print("\nTesting Parameter Generator...")
    
    # Create dummy config
    config = {
        "device": "cpu",
        "text_prompt": "Generate finetuning parameters for improved performance"
    }
    
    # Create generator
    generator = InfinityFinetuningParameterGenerator(config)
    
    # Test condition generation
    condition = generator.get_condition()
    print(f"Generated condition shape: {condition.shape}")
    
    # Test parameter generation
    target_shapes = {
        "blocks.0.attn.qkv.weight": (1024, 1024),
        "blocks.0.attn.proj.weight": (1024, 1024),
        "blocks.0.mlp.fc1.weight": (4096, 1024),
        "blocks.0.mlp.fc2.weight": (1024, 4096),
    }
    
    params = generator.generate_finetuning_parameters(target_shapes)
    print(f"Generated parameters: {len(params)} parameter tensors")
    
    # Check parameter shapes
    shape_matches = True
    for name, expected_shape in target_shapes.items():
        if name in params:
            actual_shape = params[name].shape
            if actual_shape != expected_shape:
                print(f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}")
                shape_matches = False
    
    return condition.shape[1] == 1024 and shape_matches


def test_end_to_end_generation():
    """Test end-to-end parameter generation"""
    print("\nTesting End-to-End Parameter Generation...")
    
    # Example target shapes
    target_shapes = {
        "blocks.0.attn.qkv.weight": (1024, 1024),
        "blocks.0.attn.proj.weight": (1024, 1024),
        "blocks.0.mlp.fc1.weight": (4096, 1024),
        "blocks.0.mlp.fc2.weight": (1024, 4096),
    }
    
    # Generate parameters
    generated_params = generate_infinity_finetuning_params(
        text_prompt="Generate finetuning parameters for improved performance",
        target_shapes=target_shapes,
        device="cpu",
        num_generated=2
    )
    
    print(f"Generated {len(generated_params)} parameter sets")
    
    # Check each parameter set
    for i, params in enumerate(generated_params):
        print(f"Parameter set {i+1}: {len(params)} parameters")
        for name, param in params.items():
            print(f"  {name}: {param.shape}")
    
    return len(generated_params) == 2


def test_parameter_application():
    """Test applying generated parameters to a dummy model"""
    print("\nTesting Parameter Application...")
    
    # Create a dummy model
    class DummyInfinityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'attn': nn.ModuleDict({
                        'qkv': nn.Linear(1024, 1024),
                        'proj': nn.Linear(1024, 1024),
                    }),
                    'mlp': nn.ModuleDict({
                        'fc1': nn.Linear(1024, 4096),
                        'fc2': nn.Linear(4096, 1024),
                    })
                }) for _ in range(2)
            ])
    
    model = DummyInfinityModel()
    
    # Generate parameters
    target_shapes = {
        "blocks.0.attn.qkv.weight": (1024, 1024),
        "blocks.0.attn.proj.weight": (1024, 1024),
        "blocks.0.mlp.fc1.weight": (4096, 1024),
        "blocks.0.mlp.fc2.weight": (1024, 4096),
    }
    
    generated_params = generate_infinity_finetuning_params(
        text_prompt="Generate finetuning parameters for improved performance",
        target_shapes=target_shapes,
        device="cpu",
        num_generated=1
    )[0]
    
    # Apply parameters to model
    original_params = {}
    for name, param in model.named_parameters():
        if name in generated_params:
            original_params[name] = param.data.clone()
            param.data = generated_params[name]
    
    print(f"Applied {len(generated_params)} parameters to model")
    
    # Test forward pass
    dummy_input = torch.randn(1, 10, 1024)
    try:
        output = model(dummy_input)
        print(f"Model forward pass successful, output shape: {output.shape}")
        success = True
    except Exception as e:
        print(f"Model forward pass failed: {e}")
        success = False
    
    # Restore original parameters
    for name, param in model.named_parameters():
        if name in original_params:
            param.data = original_params[name]
    
    return success


def main():
    """Run all tests"""
    print("Infinity Finetuning Parameter Generation System - Test Suite")
    print("=" * 70)
    
    tests = [
        ("LoRA Tokenizer", test_lora_tokenizer),
        ("Recurrent Module", test_recurrent_module),
        ("Encoders", test_encoders),
        ("Parameter Generator", test_parameter_generator),
        ("End-to-End Generation", test_end_to_end_generation),
        ("Parameter Application", test_parameter_application),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: ✗ ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Infinity finetuning parameter generation system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 