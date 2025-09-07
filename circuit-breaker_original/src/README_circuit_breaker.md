# Infinity Circuit Breaker Training

This directory contains the modified Infinity training scripts with circuit breaker functionality integrated.

## Overview

The circuit breaker training implements a dual-objective loss function:

1. **Retain Loss**: Ensures the fine-tuned model maintains similar behavior to the original model for normal (sanitized) prompts using L2 distance between hidden states.

2. **Circuit Breaker Loss**: Encourages the fine-tuned model to produce different representations for harmful prompts using inner product dissimilarity.

3. **Validation Monitoring**: Observes model behavior on validation data without affecting the loss.

## Key Files

### Modified Training Scripts

- `train_infinity_circuit_breaker.py`: Main training script with circuit breaker integration
- `trainer.py`: Modified trainer with circuit breaker loss support

### Circuit Breaker Components

- `lorra_circuit_breaker_new.py`: Circuit breaker loss function and data collator for Infinity models
- `cb_train_dataset_infinity.py`: Dataset class for circuit breaker training (text-only prompts)
- `test_circuit_breaker.py`: Test script to verify functionality

## Usage

### Basic Training

```bash
python train_infinity_circuit_breaker.py \
    --task_type circuit_breaker \
    --circuit_breaker_alpha 0.1 \
    --num_examples 1000 \
    --harmful_prompts_path /path/to/harmful/prompts \
    --sanitized_prompts_path /path/to/sanitized/prompts \
    --category your_category
```

### Key Arguments

- `--task_type`: Set to 'circuit_breaker' for circuit breaker training
- `--circuit_breaker_alpha`: Coefficient for circuit breaker loss (default: 0.1)
- `--circuit_breaker_target_layers`: Target layers for circuit breaker (default: "0,1,2,3,4,5")
- `--num_examples`: Number of training examples
- `--harmful_prompts_path`: Path to harmful prompts file
- `--sanitized_prompts_path`: Path to sanitized prompts file
- `--category`: Category for prompt selection
- `--validation_ratio`: Ratio of data for validation (default: 0.1)

## Loss Function Details

### Retain Loss
- **Objective**: Minimize L2 distance between fine-tuned and original model hidden states
- **Target**: Sanitized prompts
- **Formula**: `||h_finetuned - h_original||_2`

### Circuit Breaker Loss
- **Objective**: Maximize dissimilarity between fine-tuned and original model hidden states
- **Target**: Harmful prompts
- **Formula**: `ReLU(inner_product(normalized_h_finetuned, normalized_h_original))`

### Training Progress Scheduling
- **Early Training**: Higher retain loss weight, lower circuit breaker loss weight
- **Late Training**: Lower retain loss weight, higher circuit breaker loss weight
- **Formula**: `retain_coeff = alpha * progress`, `cb_coeff = alpha * (1 - progress)`

## Dataset Format

The circuit breaker dataset returns dictionaries with the following structure:

```python
{
    'text_cond_tuple': (kv_compact, lens, cu_seqlens_k, Ltext),  # Retain text
    'text_cond_tuple_circuit_breaker': (kv_compact, lens, cu_seqlens_k, Ltext),  # CB text
    'text_cond_tuple_val': (kv_compact, lens, cu_seqlens_k, Ltext),  # Validation text
    'input_ids': tensor,  # Retain input ids
    'attention_mask': tensor,  # Retain attention mask
    'input_ids_circuit_breaker': tensor,  # CB input ids
    'attention_mask_circuit_breaker': tensor,  # CB attention mask
    'input_ids_val': tensor,  # Validation input ids
    'attention_mask_val': tensor,  # Validation attention mask
}
```

## Model Compatibility

The circuit breaker loss function is designed to work with different Infinity model architectures:

- Models with `blocks` attribute and `output_hidden_states`
- Models with `unregistered_blocks` attribute and `get_hidden_states()`

## Testing

Run the test script to verify functionality:

```bash
python test_circuit_breaker.py
```

## Integration with Original Training

The modifications maintain backward compatibility with the original Infinity training:

- If `task_type` is not 'circuit_breaker', training proceeds normally
- Circuit breaker loss is only computed when data is in the expected format
- Original training loss is preserved and combined with circuit breaker loss

## Monitoring

The training script provides detailed logging:

- Progress scheduling coefficients
- Retain and circuit breaker loss values
- Cosine similarity measurements for retain, circuit breaker, and validation data
- Activation norms for monitoring model behavior

## Notes

1. The circuit breaker training uses dummy image tokens since the dataset only contains text prompts
2. Hidden state extraction is model-architecture dependent and may need adjustment for different models
3. The loss function includes proper masking for attention weights
4. Validation observations are computed but do not affect the training loss 