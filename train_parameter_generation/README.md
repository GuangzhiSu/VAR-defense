# Infinity Finetuning Parameter Generation

This directory contains the implementation of conditional parameter generation for Infinity model finetuning, using the framework from Recurrent-Parameter-Generation.

## Overview

The system generates finetuning parameters for Infinity models using a conditional diffusion-based architecture that combines:

1. **Base Model Encoding**: Extracts structural information from the Infinity model
2. **Textual Instructions**: Processes text prompts using T5 encoder to specify target task or style
3. **Recurrent Processing**: Uses LSTM-based recurrent module for scalable parameter generation
4. **Conditional Diffusion**: Generates parameters based on fused conditions

## Key Components

### 1. InfinityModelEncoder (`infinity_encoder.py`)
- Extracts structural information from Infinity model checkpoints
- Encodes model architecture details (embed_dim, depth, num_heads, etc.)
- Converts model information to embedding vectors

### 2. EnhancedTextEncoder (`infinity_encoder.py`)
- Uses T5 model for text encoding
- Processes textual instructions into embedding vectors
- Includes fallback mechanism when T5 is not available

### 3. LoRATokenizer (`conditional_generate.py`)
- Tokenizes parameter matrices into fixed-size tokens
- Adds positional annotations (layer index and intra-layer offset)
- Implements sinusoidal positional embeddings

### 4. RecurrentPrototypeModule (`conditional_generate.py`)
- LSTM-based recurrent module for processing tokenized parameters
- Generates prototype vectors that summarize local correlations
- Enables scalable processing of large parameter sets

### 5. InfinityFinetuningParameterGenerator (`conditional_generate.py`)
- Main parameter generation model
- Combines base model and text encodings
- Generates finetuning parameters for Infinity models

## Usage

### Basic Usage

```bash
# Generate parameters with default settings
python conditional_generate.py

# Generate parameters with custom text prompt
python run_infinity_finetuning_generation.py --text_prompt "Generate finetuning parameters for improved performance"

# Generate parameters with Infinity model path
python run_infinity_finetuning_generation.py \
    --infinity_model_path "/path/to/infinity/model.pth" \
    --text_prompt "Generate finetuning parameters for robustness" \
    --num_generated 10
```

### Using Preset Prompts

```bash
# Run with preset prompts for different use cases
python run_infinity_finetuning_generation.py
```

This will generate parameters for:
- **performance**: Improved performance and accuracy
- **robustness**: Enhanced robustness against adversarial attacks
- **efficiency**: Optimized computational efficiency
- **privacy**: User privacy protection
- **fairness**: Fair and unbiased model behavior
- **security**: Enhanced security against malicious inputs
- **adaptability**: Improved adaptability to new domains
- **interpretability**: Enhanced interpretability and explainability

### Advanced Usage

```bash
# Generate with custom configuration
python run_infinity_finetuning_generation.py \
    --infinity_model_path "/path/to/infinity/model.pth" \
    --text_prompt "Generate finetuning parameters for improved performance" \
    --num_generated 5 \
    --output_dir "./my_finetuning_params" \
    --device "cuda" \
    --save_config
```

## Configuration

### Command Line Arguments

- `--infinity_model_path`: Path to Infinity model checkpoint (optional)
- `--text_prompt`: Text prompt for conditional generation
- `--num_generated`: Number of parameter sets to generate (default: 5)
- `--output_dir`: Output directory for generated parameters (default: "./generated_finetuning_params")
- `--device`: Device to use for generation (default: "cuda")
- `--save_config`: Save generation configuration

### Environment Variables

```bash
export INFINITY_MODEL_PATH="/path/to/infinity/model.pth"
export DEFAULT_TEXT_PROMPT="Generate finetuning parameters for improved performance"
export DEFAULT_NUM_GENERATED=10
```

## Output Structure

The system generates the following files:

```
generated_finetuning_params/
├── target_shapes.json              # Parameter shapes extracted from model
├── generation_config.json          # Generation configuration (if --save_config)
├── finetuning_params_001.pth       # Generated parameter set 1
├── finetuning_params_002.pth       # Generated parameter set 2
└── ...
```

### Parameter Format

Generated parameters are saved as PyTorch tensors with the following structure:

```python
{
    "blocks.0.attn.qkv.weight": torch.Tensor(1024, 1024),
    "blocks.0.attn.proj.weight": torch.Tensor(1024, 1024),
    "blocks.0.mlp.fc1.weight": torch.Tensor(4096, 1024),
    "blocks.0.mlp.fc2.weight": torch.Tensor(1024, 4096),
    # ... more parameters
}
```

## Applying Generated Parameters

```python
import torch
from conditional_generate import generate_infinity_finetuning_params

# Generate parameters
generated_params = generate_infinity_finetuning_params(
    infinity_model_path="/path/to/infinity/model.pth",
    text_prompt="Generate finetuning parameters for improved performance",
    num_generated=1
)[0]

# Load your Infinity model
model = load_infinity_model("/path/to/infinity/model.pth")

# Apply generated parameters
for name, param in model.named_parameters():
    if name in generated_params:
        param.data = generated_params[name]

# Now your model has the generated finetuning parameters
```

## Testing

Run the test suite to verify the system:

```bash
python test_infinity_finetuning.py
```

This will test:
- LoRA tokenizer functionality
- Recurrent module processing
- Encoder components
- Parameter generator
- End-to-end generation
- Parameter application

## Example Text Prompts

Here are some example text prompts for different use cases:

```bash
# Performance improvement
"Generate finetuning parameters for improved performance and accuracy"

# Adversarial robustness
"Generate finetuning parameters that enhance model robustness against adversarial attacks"

# Privacy protection
"Generate finetuning parameters that protect user privacy and prevent data leakage"

# Fairness
"Generate finetuning parameters that ensure fair and unbiased model behavior"

# Efficiency
"Generate finetuning parameters that optimize computational efficiency"

# Security
"Generate finetuning parameters that enhance model security against malicious inputs"

# Domain adaptation
"Generate finetuning parameters that improve model adaptability to new domains"

# Interpretability
"Generate finetuning parameters that enhance model interpretability and explainability"
```

## Architecture Details

### Parameter Generation Process

1. **Model Analysis**: Extract structural information from Infinity model
2. **Text Encoding**: Process text prompt using T5 encoder
3. **Condition Fusion**: Combine model and text encodings
4. **Parameter Generation**: Generate parameter vector using MLP network
5. **Shape Reshaping**: Reshape parameters to match target model architecture

### Tokenization Process

1. **Flatten and Normalize**: Each parameter matrix is flattened and normalized
2. **Token Splitting**: Flattened vector is split into fixed-size tokens (default: 64)
3. **Positional Annotations**: Each token is annotated with layer index and offset
4. **Sinusoidal Embeddings**: Positional information is encoded using sinusoidal embeddings

### Recurrent Processing

1. **Token Sequence**: Tokens are processed through LSTM-based recurrent module
2. **Prototype Generation**: The module generates prototype vectors that summarize correlations
3. **Compact Representation**: The entire parameter set is encoded into a compact representation

## Dependencies

Required packages:
```bash
pip install torch transformers numpy
```

Optional packages for full functionality:
```bash
pip install mamba-ssm  # For Mamba recurrent module
```

## Troubleshooting

### Common Issues

1. **T5 Model Loading**: If T5 fails to load, the system will use a fallback encoding
2. **Infinity Model Path**: If no Infinity model path is provided, default shapes are used
3. **CUDA Memory**: For large models, consider reducing batch size or using CPU
4. **Parameter Shapes**: Ensure target shapes match your Infinity model architecture

### Error Messages

- `"Warning: Could not import Infinity modules"`: Infinity model will use default structure
- `"Warning: Could not load T5 model"`: Text encoding will use fallback method
- `"Shape mismatch"`: Check that target shapes match your model architecture

## Integration with Infinity Training

The generated parameters can be integrated into your Infinity training pipeline:

```python
# Load generated parameters
generated_params = torch.load("finetuning_params_001.pth")

# Apply to model during training
for name, param in model.named_parameters():
    if name in generated_params:
        # Use generated parameters as initialization or regularization
        param.data = param.data + 0.1 * generated_params[name]
```

## Future Enhancements

Potential improvements:
1. **Better Infinity Model Loading**: More robust loading from various checkpoint formats
2. **Advanced Text Encoders**: Support for other text encoders (BERT, RoBERTa, etc.)
3. **Mamba Integration**: Replace LSTM with Mamba for better efficiency
4. **Multi-modal Conditions**: Support for additional condition types (images, audio, etc.)
5. **Training Integration**: Direct integration with Infinity training pipeline 