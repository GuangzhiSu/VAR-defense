# Infinity Circuit Breaker Implementation

This document describes the adaptation of the Circuit Breaker safety mechanism for the Infinity text-to-image model.

## Overview

The Circuit Breaker mechanism has been adapted to work with Infinity's architecture, which includes:
- VAE encoder/decoder for image processing
- GPT-style transformer for image token prediction
- T5 text encoder for text conditioning
- Cross-attention mechanisms
- Dynamic resolution training

## Key Components

### 1. `lorra_circuit_breaker_infinity.py`

Main training script that implements the circuit breaker loss function for Infinity models.

**Key Features:**
- `compute_infinity_circuit_breaker_loss()`: Dual-objective loss function
- `get_infinity_generation()`: Image generation function
- `train_infinity_circuit_breaker()`: Main training loop

**Loss Components:**
- **Retain Loss**: Ensures normal image generation capability is maintained
- **Circuit Breaker Loss**: Prevents harmful content generation

### 2. `cb_train_dataset_infinity.py`

Dataset class for Infinity circuit breaker training.

**Data Sources:**
- **Retain Data**: Normal/safe prompts for maintaining generation quality
- **Circuit Breaker Data**: Harmful prompts for safety training
- **Validation Data**: Mixed prompts for evaluation

## Architecture Adaptations

### Model Integration

The circuit breaker integrates with Infinity's architecture through:

1. **Text Conditioning**: Uses T5 encoder to process text prompts
2. **Image Tokenization**: Uses VAE to encode/decode images
3. **Dynamic Resolution**: Supports multiple image resolutions
4. **Cross-Attention**: Leverages Infinity's cross-attention for text-image fusion

### Loss Function Design

```python
def compute_infinity_circuit_breaker_loss(self, model, inputs, target_layers, alpha, return_outputs=False):
    # Extract inputs for Infinity model
    text_cond_tuple = inputs.get("text_cond_tuple")
    image_tokens = inputs.get("image_tokens")
    scale_schedule = inputs.get("scale_schedule")
    
    # Progressive training coefficients
    progress = self.get_training_progress()
    retain_coeff, circuit_breaker_coeff = alpha * progress, alpha * (1-progress)
    
    # Retain loss: maintain normal generation
    retain_loss = compute_retain_loss(model, text_cond_tuple, image_tokens, scale_schedule)
    
    # Circuit breaker loss: prevent harmful generation
    circuit_breaker_loss = compute_circuit_breaker_loss(model, cb_inputs)
    
    return retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss
```

## Usage

### 1. Setup

```bash
# Install dependencies
pip install transformers torch peft

# Set up Infinity model path
export INFINITY_MODEL_PATH="/path/to/infinity/weights"
export VAE_CHECKPOINT="/path/to/vae/checkpoint"
```

### 2. Training

```bash
python lorra_circuit_breaker_infinity.py \
    --model_name_or_path $INFINITY_MODEL_PATH \
    --vae_ckpt $VAE_CHECKPOINT \
    --target_layers "0,1,2,3,4,5" \
    --transform_layers "0,1,2,3,4,5" \
    --lorra_alpha 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_dir "./infinity_circuit_breaker_output" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_grad_norm 0.3 \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 100
```

### 3. Evaluation

The training script includes evaluation functions that test:
- Normal image generation quality
- Harmful prompt rejection
- Safety mechanism effectiveness

## Key Differences from LLM Circuit Breaker

### 1. Input Format
- **LLM**: Text tokens and attention masks
- **Infinity**: Text condition tuples, image tokens, and scale schedules

### 2. Model Architecture
- **LLM**: Causal language model with self-attention
- **Infinity**: Transformer with cross-attention between text and image tokens

### 3. Loss Computation
- **LLM**: Hidden state comparison in text space
- **Infinity**: Hidden state comparison in image token space

### 4. Generation Process
- **LLM**: Autoregressive text generation
- **Infinity**: Autoregressive image token generation followed by VAE decoding

## Safety Mechanisms

### 1. Content Safety Detection
The circuit breaker can be enhanced with:
- Image content safety classifiers
- NSFW detection models
- Violence detection systems

### 2. Prompt Filtering
- Real-time harmful prompt detection
- Safe response generation
- Content moderation integration

## Configuration

### Model Parameters
```python
# Infinity model configuration
model_config = {
    'embed_dim': 2048,
    'depth': 32,
    'num_heads': 16,
    'mlp_ratio': 4.0,
    'text_channels': 2048,
    'use_bit_label': 1,
    'rope2d_each_sa_layer': 1,
    'add_lvl_embeding_only_first_block': 1
}
```

### Circuit Breaker Parameters
```python
# Circuit breaker configuration
cb_config = {
    'target_layers': [0, 1, 2, 3, 4, 5],  # Layers to apply circuit breaker
    'lorra_alpha': 0.1,  # Loss balancing coefficient
    'progressive_training': True,  # Gradually shift focus
    'safety_threshold': 0.8  # Safety detection threshold
}
```

## Limitations and Considerations

### 1. Computational Requirements
- Infinity models are large and require significant GPU memory
- Training may need distributed training strategies
- VAE encoding/decoding adds computational overhead

### 2. Data Requirements
- Need paired text-image data for training
- Harmful content datasets must be carefully curated
- Safety evaluation requires human annotation

### 3. Model Compatibility
- Circuit breaker is designed for specific Infinity model versions
- May need adaptation for different model sizes
- VAE compatibility must be ensured

## Future Improvements

### 1. Enhanced Safety Detection
- Integrate multiple safety classifiers
- Real-time content filtering
- Adaptive safety thresholds

### 2. Better Training Strategies
- Curriculum learning for progressive safety training
- Adversarial training for robustness
- Multi-objective optimization

### 3. Evaluation Metrics
- Automated safety evaluation
- Quality preservation metrics
- User preference alignment

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use gradient checkpointing
2. **VAE Compatibility**: Ensure VAE checkpoint matches model version
3. **Text Encoding**: Verify T5 encoder compatibility
4. **Scale Schedule**: Check dynamic resolution configuration

### Debugging Tips

1. Start with smaller models for testing
2. Use simplified datasets initially
3. Monitor loss components separately
4. Validate text encoding pipeline

## References

- Original Circuit Breaker paper
- Infinity model architecture documentation
- T5 text encoding implementation
- VAE image processing details 