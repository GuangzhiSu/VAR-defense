import sys, os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Import our custom encoders
from infinity_encoder import InfinityModelEncoder, EnhancedTextEncoder, create_condition_fusion

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LoRATokenizer:
    """Tokenize LoRA matrices into fixed-size tokens with positional annotations"""
    def __init__(self, token_size=64):
        self.token_size = token_size
        
    def lora_matrix_to_tokens(self, lora_matrix, layer_idx=0):
        """Convert LoRA matrix to tokens with positional annotations"""
        # Flatten the matrix
        flattened = lora_matrix.flatten()
        
        # Normalize per layer
        flattened = F.normalize(flattened, p=2, dim=0)
        
        # Split into tokens
        num_tokens = (flattened.shape[0] + self.token_size - 1) // self.token_size
        tokens = []
        
        for i in range(num_tokens):
            start_idx = i * self.token_size
            end_idx = min(start_idx + self.token_size, flattened.shape[0])
            token = flattened[start_idx:end_idx]
            
            # Pad if necessary
            if token.shape[0] < self.token_size:
                padding = torch.zeros(self.token_size - token.shape[0], device=token.device)
                token = torch.cat([token, padding])
            
            # Add positional annotation (layer index and intra-layer offset)
            pos_embedding = self._get_positional_embedding(layer_idx, i, token.device)
            token_with_pos = torch.cat([token, pos_embedding])
            
            tokens.append(token_with_pos)
        
        return torch.stack(tokens)
    
    def _get_positional_embedding(self, layer_idx, token_idx, device):
        """Generate sinusoidal positional embedding"""
        # Simple sinusoidal embedding for layer and token position
        pos = layer_idx * 1000 + token_idx  # Combine layer and token position
        embedding_dim = 32  # Size of positional embedding
        
        pos_embedding = torch.zeros(embedding_dim, device=device)
        for i in range(embedding_dim):
            if i % 2 == 0:
                pos_embedding[i] = torch.sin(pos / (10000 ** (i / embedding_dim)))
            else:
                pos_embedding[i] = torch.cos(pos / (10000 ** ((i-1) / embedding_dim)))
        
        return pos_embedding
    
    def tokens_to_lora_matrix(self, tokens, original_shape):
        """Convert tokens back to LoRA matrix"""
        # Remove positional embeddings (last 32 dimensions)
        tokens_clean = tokens[..., :-32]
        
        # Flatten tokens
        flattened = tokens_clean.flatten()
        
        # Reshape to original shape
        total_elements = np.prod(original_shape)
        if flattened.shape[0] > total_elements:
            flattened = flattened[:total_elements]
        elif flattened.shape[0] < total_elements:
            padding = torch.zeros(total_elements - flattened.shape[0], device=flattened.device)
            flattened = torch.cat([flattened, padding])
        
        return flattened.reshape(original_shape)


class RecurrentPrototypeModule(nn.Module):
    """Recurrent module for processing tokenized LoRA updates"""
    def __init__(self, token_dim=96, hidden_dim=256):  # 64 + 32 (token + pos)
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        
        # Simple LSTM-based recurrent module
        self.lstm = nn.LSTM(
            input_size=token_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Projection to prototype dimension
        self.prototype_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, tokens, h_0=None, c_0=None):
        """Process tokens and generate prototypes"""
        # tokens shape: (batch_size, seq_len, token_dim)
        batch_size, seq_len, _ = tokens.shape
        
        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = torch.zeros(2, batch_size, self.hidden_dim, device=tokens.device)
        if c_0 is None:
            c_0 = torch.zeros(2, batch_size, self.hidden_dim, device=tokens.device)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(tokens, (h_0, c_0))
        
        # Generate prototypes
        prototypes = self.prototype_proj(lstm_out)
        
        return prototypes, (h_n, c_n)


class InfinityFinetuningParameterGenerator(nn.Module):
    """Generate finetuning parameters for Infinity model using conditional diffusion"""
    
    def __init__(self, config, infinity_model_path=None, text_prompt=None):
        super().__init__()
        self.config = config
        self.text_prompt = text_prompt or "Generate finetuning parameters for improved performance"
        
        # Initialize encoders
        self.base_encoder = InfinityModelEncoder(embed_dim=512, model_path=infinity_model_path)
        self.text_encoder = EnhancedTextEncoder(embed_dim=512)
        self.condition_fusion = create_condition_fusion(fused_dim=1024)
        
        # Initialize tokenizer and recurrent module
        self.tokenizer = LoRATokenizer(token_size=64)
        self.recurrent_module = RecurrentPrototypeModule()
        
        # Parameter generation network
        self.parameter_generator = self._build_parameter_generator()
        
    def _build_parameter_generator(self):
        """Build the parameter generation network"""
        return nn.Sequential(
            nn.Linear(1024, 2048),  # Input: fused condition
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 8192),  # Output: parameter vector
            nn.Tanh(),  # Bound parameters
        )
        
    def get_condition(self):
        """Generate condition from base model and text prompt"""
        # Encode base model
        base_embedding = self.base_encoder()
        
        # Encode text prompt
        text_embedding = self.text_encoder(self.text_prompt)
        
        # Fuse conditions
        condition = self.condition_fusion(torch.cat([base_embedding, text_embedding], dim=-1))
        
        return condition
    
    def process_lora_tokens(self, lora_tokens):
        """Process LoRA tokens through recurrent module to get prototypes"""
        # lora_tokens shape: (batch_size, seq_len, token_dim)
        prototypes, _ = self.recurrent_module(lora_tokens)
        return prototypes
    
    def generate_finetuning_parameters(self, target_shapes=None):
        """Generate finetuning parameters for Infinity model"""
        # Get condition
        condition = self.get_condition()
        
        # Generate parameter vector
        param_vector = self.parameter_generator(condition)
        
        # If target shapes are provided, reshape accordingly
        if target_shapes is not None:
            return self._reshape_to_target_shapes(param_vector, target_shapes)
        
        return param_vector
    
    def _reshape_to_target_shapes(self, param_vector, target_shapes):
        """Reshape parameter vector to target shapes"""
        # Flatten parameter vector
        flat_params = param_vector.flatten()
        
        # Distribute parameters across target shapes
        reshaped_params = {}
        start_idx = 0
        
        for name, shape in target_shapes.items():
            num_elements = np.prod(shape)
            if start_idx + num_elements <= flat_params.shape[0]:
                param_slice = flat_params[start_idx:start_idx + num_elements]
                reshaped_params[name] = param_slice.reshape(shape)
                start_idx += num_elements
            else:
                # Pad if not enough parameters
                needed = num_elements - (flat_params.shape[0] - start_idx)
                param_slice = torch.cat([
                    flat_params[start_idx:],
                    torch.zeros(needed, device=flat_params.device)
                ])
                reshaped_params[name] = param_slice.reshape(shape)
                break
        
        return reshaped_params
    
    def forward(self, sample=False, target_shapes=None, **kwargs):
        """Forward pass with conditional generation"""
        if sample:
            return self.generate_finetuning_parameters(target_shapes)
        else:
            # Training mode - generate parameters
            return self.generate_finetuning_parameters(target_shapes)


def generate_infinity_finetuning_params(
    infinity_model_path=None,
    text_prompt="Generate finetuning parameters for improved performance",
    target_shapes=None,
    device="cuda",
    num_generated=1
):
    """Generate finetuning parameters for Infinity model"""
    
    config = {
        "device": device,
        "text_prompt": text_prompt,
        "infinity_model_path": infinity_model_path
    }
    
    # Create generator
    generator = InfinityFinetuningParameterGenerator(config)
    generator = generator.to(device)
    generator.eval()
    
    # Generate parameters
    generated_params = []
    
    with torch.no_grad():
        for i in range(num_generated):
            print(f"Generating parameter set {i+1}/{num_generated}...")
            params = generator(sample=True, target_shapes=target_shapes)
            generated_params.append(params)
    
    return generated_params


def save_finetuning_params(params, save_path):
    """Save generated finetuning parameters"""
    torch.save(params, save_path)
    print(f"Parameters saved to: {save_path}")


# Example usage and testing
if __name__ == "__main__":
    # Example target shapes for Infinity model parameters
    example_target_shapes = {
        "blocks.0.attn.qkv.weight": (1024, 1024),
        "blocks.0.attn.proj.weight": (1024, 1024),
        "blocks.0.mlp.fc1.weight": (4096, 1024),
        "blocks.0.mlp.fc2.weight": (1024, 4096),
        "blocks.1.attn.qkv.weight": (1024, 1024),
        "blocks.1.attn.proj.weight": (1024, 1024),
        "blocks.1.mlp.fc1.weight": (4096, 1024),
        "blocks.1.mlp.fc2.weight": (1024, 4096),
    }
    
    # Generate parameters
    print("Generating Infinity finetuning parameters...")
    print(f"Text prompt: Generate finetuning parameters for improved performance")
    
    generated_params = generate_infinity_finetuning_params(
        text_prompt="Generate finetuning parameters for improved performance",
        target_shapes=example_target_shapes,
        device="cpu",  # Use CPU for testing
        num_generated=2
    )
    
    # Save parameters
    for i, params in enumerate(generated_params):
        save_path = f"infinity_finetuning_params_{i+1}.pth"
        save_finetuning_params(params, save_path)
    
    print("Parameter generation completed!") 