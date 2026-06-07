# ELG 5214 – Assignment 1  
## Performance Comparison of JAX and PyTorch on MNIST


## Overview

This project implements and compares JAX and PyTorch for training an MLP on the MNIST digit classification dataset.

Both implementations use identical architecture and optimization settings to ensure a fair comparison.


## Model Architecture

The neural network architecture is:

784 → 256 → 128 → 10

- ReLU activations after the first two layers
- Cross-entropy loss
- Adam optimizer (learning rate = 1e-3)
- 50 training epochs

Dataset split:

- 50,000 training samples  
- 10,000 validation samples  
- 10,000 test samples  


## Files
https://colab.research.google.com/drive/1a4VcvkAajFwaLjWYT0TXtjn30d0R0wz3?usp=sharing

## How to Run (Google Colab)

1. Open the notebook in Google Colab.
2. Ensure GPU runtime is enabled:
3. Run all cells from top to bottom.

## Expexcted Output

For both JAX and PyTorch, the notebook produces:

- Performance summary tables
- Steady-state epoch timing
- Final test accuracy
- Combined training loss plots
- Combined validation accuracy plots

Results are saved into:

- `outputs_jax/`
- `outputs_torch/`

