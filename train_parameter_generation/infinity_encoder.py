import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
from pathlib import Path

# Add Infinity path
infinity_path = os.path.join(os.path.dirname(__file__), "../infinity")
sys.path.append(infinity_path)

try:
    from infinity.models.infinity import Infinity
    from infinity.models.t5 import T5TextEncoder
    from infinity.models.bsq_vae import BSQVAE
    from run_infinity import load_visual_tokenizer, load_transformer, load_tokenizer
except ImportError as e:
    print(f"Warning: Could not import Infinity modules: {e}")
    print("Using dummy model for base model encoding")


class InfinityModelEncoder(nn.Module):
    """Encoder for Infinity model structural information using string-based encoding"""
    
    def __init__(self, embed_dim=512, model_path=None, use_bert_encoder=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_path = model_path
        self.use_bert_encoder = use_bert_encoder
        
        # Initialize string encoder
        if use_bert_encoder:
            self.string_encoder = self._init_bert_encoder()
        else:
            self.string_encoder = self._init_simple_encoder()
        
        # Load Infinity model if path is provided
        self.infinity_model = None
        if model_path and os.path.exists(model_path):
            self.infinity_model = self.load_infinity_model(model_path)
    
    def _init_bert_encoder(self):
        """Initialize BERT-based string encoder"""
        try:
            from transformers import BertTokenizer, BertModel
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_loaded = True
            
            # Projection layer to map BERT output to desired dimension
            return nn.Sequential(
                nn.Linear(768, self.embed_dim * 2),  # BERT hidden size is 768
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
            self.bert_loaded = False
            return self._init_simple_encoder()
    
    def _init_simple_encoder(self):
        """Initialize simple MLP-based string encoder"""
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
    
    def load_infinity_model(self, model_path):
        """Load Infinity model from checkpoint"""
        try:
            # This is a simplified loading - you may need to adjust based on your setup
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model configuration from checkpoint
            model_config = self.extract_config_from_checkpoint(checkpoint)
            
            # Create a dummy model with the extracted config
            class DummyInfinityModel:
                def __init__(self, config):
                    self.C = config.get('embed_dim', 1024)
                    self.depth = config.get('depth', 16)
                    self.num_heads = config.get('num_heads', 16)
                    self.mlp_ratio = config.get('mlp_ratio', 4.0)
                    self.codebook_dim = config.get('codebook_dim', 56)
                    self.V = config.get('vocab_size', 8192)
                    self.Ct5 = config.get('text_channels', 2048)
                    self.config = config
            
            return DummyInfinityModel(model_config)
            
        except Exception as e:
            print(f"Warning: Could not load Infinity model from {model_path}: {e}")
            return self.create_default_infinity_model()
    
    def extract_config_from_checkpoint(self, checkpoint):
        """Extract configuration from checkpoint"""
        config = {}
        
        # Try to extract from state dict keys
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Analyze layer structure
        layer_count = 0
        embed_dim = None
        num_heads = None
        
        for key in state_dict.keys():
            if 'blocks.' in key and '.mlp.' in key:
                layer_count = max(layer_count, int(key.split('.')[1]) + 1)
            elif 'blocks.0.attn.qkv.weight' in key:
                embed_dim = state_dict[key].shape[0]
            elif 'blocks.0.attn.proj.weight' in key:
                if embed_dim is None:
                    embed_dim = state_dict[key].shape[1]
        
        config.update({
            'embed_dim': embed_dim or 1024,
            'depth': layer_count or 16,
            'num_heads': num_heads or 16,
            'mlp_ratio': 4.0,
            'codebook_dim': 56,
            'vocab_size': 8192,
            'text_channels': 2048,
        })
        
        return config
    
    def create_default_infinity_model(self):
        """Create default Infinity model structure"""
        class DefaultInfinityModel:
            def __init__(self):
                self.C = 1024
                self.depth = 16
                self.num_heads = 16
                self.mlp_ratio = 4.0
                self.codebook_dim = 56
                self.V = 8192
                self.Ct5 = 2048
                self.config = {
                    'embed_dim': 1024,
                    'depth': 16,
                    'num_heads': 16,
                    'mlp_ratio': 4.0,
                    'codebook_dim': 56,
                    'vocab_size': 8192,
                    'text_channels': 2048,
                }
        
        return DefaultInfinityModel()
    
    def extract_model_info(self, model=None):
        """Extract structural information from Infinity model"""
        if model is None:
            model = self.infinity_model or self.create_default_infinity_model()
        
        # Extract key structural information
        info = {
            'embed_dim': getattr(model, 'C', 1024),
            'depth': getattr(model, 'depth', 16),
            'num_heads': getattr(model, 'num_heads', 16),
            'mlp_ratio': getattr(model, 'mlp_ratio', 4.0),
            'codebook_dim': getattr(model, 'codebook_dim', 56),
            'vocab_size': getattr(model, 'V', 8192),
            'text_channels': getattr(model, 'Ct5', 2048),
        }
        
        # Add additional structural information if available
        if hasattr(model, 'config'):
            config = model.config
            info.update({
                'model_type': config.get('model_type', 'infinity'),
                'rope2d_each_sa_layer': config.get('rope2d_each_sa_layer', 0),
                'rope2d_normalized_by_hw': config.get('rope2d_normalized_by_hw', 0),
                'use_bit_label': config.get('use_bit_label', 1),
                'add_lvl_embeding_only_first_block': config.get('add_lvl_embeding_only_first_block', 1),
            })
        
        return info
    
    def flatten_structural_metadata_to_string(self, model_info):
        """Flatten structural metadata and convert to string representation"""
        # Step 1: Flatten structural metadata
        structural_data = [
            f"embed_dim:{model_info['embed_dim']}",
            f"depth:{model_info['depth']}",
            f"num_heads:{model_info['num_heads']}",
            f"mlp_ratio:{model_info['mlp_ratio']}",
            f"codebook_dim:{model_info['codebook_dim']}",
            f"vocab_size:{model_info['vocab_size']}",
            f"text_channels:{model_info['text_channels']}",
        ]
        
        # Add additional metadata if available
        if 'model_type' in model_info:
            structural_data.append(f"model_type:{model_info['model_type']}")
        if 'rope2d_each_sa_layer' in model_info:
            structural_data.append(f"rope2d_each_sa_layer:{model_info['rope2d_each_sa_layer']}")
        if 'rope2d_normalized_by_hw' in model_info:
            structural_data.append(f"rope2d_normalized_by_hw:{model_info['rope2d_normalized_by_hw']}")
        if 'use_bit_label' in model_info:
            structural_data.append(f"use_bit_label:{model_info['use_bit_label']}")
        if 'add_lvl_embeding_only_first_block' in model_info:
            structural_data.append(f"add_lvl_embeding_only_first_block:{model_info['add_lvl_embeding_only_first_block']}")
        
        # Step 2: Convert to string
        structural_string = "_".join(structural_data)
        
        return structural_string
    
    def encode_string_to_embedding(self, structural_string):
        """Encode structural string to embedding vector"""
        if self.use_bert_encoder and hasattr(self, 'bert_loaded') and self.bert_loaded:
            return self._encode_with_bert(structural_string)
        else:
            return self._encode_with_hash(structural_string)
    
    def _encode_with_bert(self, structural_string):
        """Encode using BERT"""
        # Tokenize the structural string
        inputs = self.bert_tokenizer(
            structural_string,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode with BERT
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use mean pooling over sequence length
            bert_embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Project to desired dimension
        encoded_embedding = self.string_encoder(bert_embedding)
        
        return encoded_embedding
    
    def _encode_with_hash(self, structural_string):
        """Encode using hash-based method"""
        # Use hash-based encoding for the string
        hash_val = hash(structural_string) % (2**31)
        torch.manual_seed(hash_val)
        
        # Generate initial embedding from string
        device = next(self.parameters()).device
        initial_embedding = torch.randn(self.embed_dim, device=device)
        
        # Pass through string encoder
        encoded_embedding = self.string_encoder(initial_embedding.unsqueeze(0))
        
        return encoded_embedding
    
    def forward(self, model=None):
        """Forward pass: extract model info, flatten to string, and encode to embedding"""
        # Step 1: Extract structural information
        model_info = self.extract_model_info(model)
        
        # Step 2: Flatten structural metadata to string
        structural_string = self.flatten_structural_metadata_to_string(model_info)
        
        # Step 3: Encode string to embedding
        embedding = self.encode_string_to_embedding(structural_string)
        
        return embedding


class EnhancedTextEncoder(nn.Module):
    """Enhanced text encoder with better T5 integration"""
    
    def __init__(self, model_name="t5-base", embed_dim=512, max_length=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        try:
            from transformers import T5Tokenizer, T5EncoderModel
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.t5_encoder = T5EncoderModel.from_pretrained(model_name)
            self.t5_loaded = True
        except Exception as e:
            print(f"Warning: Could not load T5 model: {e}")
            self.t5_loaded = False
        
        # Projection layer
        if self.t5_loaded:
            hidden_size = self.t5_encoder.config.hidden_size
        else:
            hidden_size = 768  # Default size
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Positional encoding for fallback
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
    
    def forward(self, text):
        """Encode text to embedding"""
        if self.t5_loaded:
            return self._encode_with_t5(text)
        else:
            return self._encode_fallback(text)
    
    def _encode_with_t5(self, text):
        """Encode using T5"""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.t5_encoder(**inputs)
            # Use mean pooling over sequence length
            text_embedding = outputs.last_hidden_state.mean(dim=1)
        
        return self.projection(text_embedding)
    
    def _encode_fallback(self, text):
        """Fallback encoding when T5 is not available"""
        # Simple hash-based encoding
        hash_val = hash(text) % (2**31)
        torch.manual_seed(hash_val)
        
        # Generate random embedding
        device = next(self.parameters()).device
        embedding = torch.randn(1, self.embed_dim, device=device)
        
        return self.projection(embedding)


def create_condition_fusion(base_embed_dim=512, text_embed_dim=512, fused_dim=1024):
    """Create condition fusion module"""
    return nn.Sequential(
        nn.Linear(base_embed_dim + text_embed_dim, fused_dim),
        nn.LayerNorm(fused_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(fused_dim, fused_dim // 2),
        nn.LayerNorm(fused_dim // 2),
        nn.GELU(),
        nn.Linear(fused_dim // 2, fused_dim),
        nn.LayerNorm(fused_dim),
    )


# Test function
def test_encoders():
    """Test the encoders"""
    print("Testing Infinity Model Encoder...")
    
    # Test base model encoder with BERT
    base_encoder = InfinityModelEncoder(embed_dim=512, use_bert_encoder=True)
    base_embedding = base_encoder()
    print(f"Base embedding shape: {base_embedding.shape}")
    
    # Test base model encoder with simple encoder
    base_encoder_simple = InfinityModelEncoder(embed_dim=512, use_bert_encoder=False)
    base_embedding_simple = base_encoder_simple()
    print(f"Base embedding (simple) shape: {base_embedding_simple.shape}")
    
    # Test text encoder
    text_encoder = EnhancedTextEncoder(embed_dim=512)
    text_embedding = text_encoder("Generate parameters that are robust against adversarial attacks")
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Test fusion
    fusion = create_condition_fusion()
    condition = fusion(torch.cat([base_embedding, text_embedding], dim=-1))
    print(f"Fused condition shape: {condition.shape}")
    
    print("All encoders working correctly!")


if __name__ == "__main__":
    test_encoders() 