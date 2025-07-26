# Harmfulness Probe for Infinity Model

This script implements a harmfulness detection probe for the Infinity model, transformed from a Jupyter notebook to a Python script.

## Overview

The script performs the following tasks:

1. **Data Preparation**: Loads harmful prompts and creates sanitized versions using an LLM
2. **Model Loading**: Loads the Infinity model, VAE, and text encoder
3. **Feature Extraction**: Uses hooks to extract hidden layer representations
4. **Training**: Trains an MLP classifier to detect harmful content
5. **Testing**: Evaluates the probe on both harmful and normal datasets

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm
- datasets

## Usage

```bash
cd /home/gs285/VAR/my_model/harmfulness_probe
python harmfulness_probe.py
```

## Configuration

The script uses the following key parameters:

- `probe_type`: "mlp" (Multi-Layer Perceptron classifier)
- `select_layer`: 30 (which layer to extract features from)
- `threshold`: 0.99 (detection threshold)
- `pos_size`: 1200 (number of positive examples)
- `neg_size`: 2400 (number of negative examples)

## File Structure

- `harmfulness_probe.py`: Main script
- `README.md`: This file

## Output

The script saves:
- Trained model: `/home/gs285/VAR/Infinity/new_dataset/saved_probe/mlp_probe_model.joblib`
- Training data: `/home/gs285/VAR/Infinity/new_dataset/saved_probe/x_train.npy` and `y_train.npy`
- Test results: `/home/gs285/VAR/Infinity/new_dataset/cors.json`

## Notes

- The script requires CUDA GPU support
- Make sure all model weights are available in the specified paths
- The script includes error handling for CUDA device availability issues 