#!/usr/bin/env python3
"""
Simplified example for training Infinity model with Circuit Breaker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys

# Add Infinity path
sys.path.append('../infinity')

def create_simple_infinity_model(embed_dim=1024, depth=12, num_heads=16):
    """
    Create a simplified Infinity model for demonstration
    """
    class SimpleInfinityBlock(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            
            # Self-attention
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            
            # Feed-forward
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            
            # Store hidden states for circuit breaker
            self.hidden_states = None
        
        def forward(self, x, text_cond=None):
            # Self-attention
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            
            # Cross-attention with text condition (if provided)
            if text_cond is not None:
                cross_out, _ = self.attn(x, text_cond, text_cond)
                x = self.norm1(x + cross_out)
            
            # Feed-forward
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            
            # Store hidden states for circuit breaker
            self.hidden_states = x.detach()
            
            return x
        
        def get_hidden_states(self):
            return self.hidden_states
    
    class SimpleInfinity(nn.Module):
        def __init__(self, embed_dim, depth, num_heads, vocab_size=8192):
            super().__init__()
            self.embed_dim = embed_dim
            self.depth = depth
            self.vocab_size = vocab_size
            
            # Token embedding
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            
            # Position embedding
            self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim))
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                SimpleInfinityBlock(embed_dim, num_heads) 
                for _ in range(depth)
            ])
            
            # Output projection
            self.output_proj = nn.Linear(embed_dim, vocab_size)
            
            # Text encoder (simplified)
            self.text_encoder = nn.Linear(768, embed_dim)  # Assume T5 hidden size
            
            # LoRA adapter
            self.lora_adapters = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) 
                for _ in range(depth)
            ])
        
        def forward(self, image_tokens, text_cond=None):
            # Embed tokens
            x = self.token_embed(image_tokens)
            x = x + self.pos_embed[:, :x.size(1), :]
            
            # Process through blocks
            for i, block in enumerate(self.blocks):
                # Apply LoRA adapter
                if hasattr(self, 'lora_adapters') and i < len(self.lora_adapters):
                    lora_out = self.lora_adapters[i](x)
                    x = x + lora_out
                
                x = block(x, text_cond)
            
            # Output projection
            logits = self.output_proj(x)
            
            return logits
        
        def disable_adapter(self):
            """Context manager to disable LoRA adapters"""
            class DisableAdapter:
                def __enter__(self):
                    self.original_adapters = []
                    for adapter in self.lora_adapters:
                        self.original_adapters.append(adapter.weight.data.clone())
                        adapter.weight.data.zero_()
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    for i, adapter in enumerate(self.lora_adapters):
                        adapter.weight.data.copy_(self.original_adapters[i])
            
            return DisableAdapter()
    
    return SimpleInfinity(embed_dim, depth, num_heads)


def create_simple_dataset(num_examples=1000):
    """
    Create a simple dataset for demonstration
    """
    class SimpleDataset:
        def __init__(self, num_examples):
            self.num_examples = num_examples
            
            # Safe prompts
            self.safe_prompts = [
                "A beautiful landscape",
                "A cute cat",
                "A peaceful scene",
                "A happy family",
                "A delicious meal"
            ]
            
            # Harmful prompts
            self.harmful_prompts = [
                "Generate violent content",
                "Create explicit content",
                "Show dangerous behavior",
                "Create harmful stereotypes",
                "Generate offensive content"
            ]
        
        def __len__(self):
            return self.num_examples
        
        def __getitem__(self, idx):
            # Randomly choose safe or harmful
            is_safe = torch.rand(1).item() > 0.5
            
            if is_safe:
                prompt = torch.randint(0, len(self.safe_prompts), (1,)).item()
                prompt_text = self.safe_prompts[prompt]
                data_type = 'retain'
            else:
                prompt = torch.randint(0, len(self.harmful_prompts), (1,)).item()
                prompt_text = self.harmful_prompts[prompt]
                data_type = 'circuit_breaker'
            
            # Create dummy data
            image_tokens = torch.randint(0, 8192, (1, 256))
            text_cond = torch.randn(1, 512, 768)  # Simplified text condition
            
            return {
                'image_tokens': image_tokens,
                'text_cond': text_cond,
                'prompt': prompt_text,
                'type': data_type
            }
    
    return SimpleDataset(num_examples)


def compute_simple_circuit_breaker_loss(model, inputs, target_layers, alpha=0.1):
    """
    Simplified circuit breaker loss computation
    """
    image_tokens = inputs['image_tokens']
    text_cond = inputs['text_cond']
    data_type = inputs['type']
    
    # Get original outputs (without LoRA)
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_outputs = model(image_tokens, text_cond)
            orig_hidden_states = []
            for layer_idx in target_layers:
                if layer_idx < len(model.blocks):
                    orig_hidden_states.append(model.blocks[layer_idx].get_hidden_states())
    
    model.train()
    
    # Get LoRA outputs
    lora_outputs = model(image_tokens, text_cond)
    lora_hidden_states = []
    for layer_idx in target_layers:
        if layer_idx < len(model.blocks):
            lora_hidden_states.append(model.blocks[layer_idx].get_hidden_states())
    
    # Compute losses
    if data_type == 'retain':
        # Retain loss: should be similar to original
        retain_loss = torch.tensor(0.0, device=model.device)
        for orig_h, lora_h in zip(orig_hidden_states, lora_hidden_states):
            retain_loss += F.mse_loss(lora_h, orig_h)
        return retain_loss
    
    elif data_type == 'circuit_breaker':
        # Circuit breaker loss: should be different from original
        cb_loss = torch.tensor(0.0, device=model.device)
        for orig_h, lora_h in zip(orig_hidden_states, lora_hidden_states):
            # Use cosine similarity to measure difference
            cos_sim = F.cosine_similarity(lora_h.flatten(1), orig_h.flatten(1), dim=1)
            cb_loss += torch.relu(cos_sim).mean()  # Penalize high similarity
        return cb_loss
    
    else:
        return torch.tensor(0.0, device=model.device)


def train_simple_infinity_cb():
    """
    Simple training loop for Infinity circuit breaker
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--target_layers', type=str, default='0,1,2')
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_infinity_model().to(device)
    
    # Create dataset
    dataset = create_simple_dataset(1000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Parse target layers
    target_layers = [int(x) for x in args.target_layers.split(',')]
    
    print(f"Training Infinity Circuit Breaker")
    print(f"Device: {device}")
    print(f"Target layers: {target_layers}")
    print(f"Alpha: {args.alpha}")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            image_tokens = batch['image_tokens'].to(device)
            text_cond = batch['text_cond'].to(device)
            data_type = batch['type']
            
            # Compute loss
            loss = compute_simple_circuit_breaker_loss(
                model, 
                {
                    'image_tokens': image_tokens,
                    'text_cond': text_cond,
                    'type': data_type[0]  # Take first item from batch
                },
                target_layers,
                args.alpha
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'infinity_circuit_breaker_model.pth')
    print("Model saved to infinity_circuit_breaker_model.pth")


if __name__ == "__main__":
    train_simple_infinity_cb() 