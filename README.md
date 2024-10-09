# GPT-2 Implementation from scratch
<div align="center">
  <h3>A modular implementation of the GPT-2 language model for training and inference</h3>
  <p>
    <img src="https://img.shields.io/badge/python-3.7%2B-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/pytorch-1.9%2B-orange.svg" alt="PyTorch Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
</div>

## Table of Contents

- [Overview](#overview)
- [Model Structure](#model-structure)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Author](#author)

## Overview

This project provides a modular implementation of the GPT-2 language model, allowing for easy training and inference. It includes a configurable model architecture, data loading utilities, and scripts for both training and text generation. 
A few modern optimizations included that weren't in the original GPT-2 paper but are commonly used in implementations:
- The use of F.scaled_dot_product_attention with is_causal=True for efficient attention computation.
- Some initialization tweaks, like the NANOGPT_SCALE_INIT attribute.

These optimizations don't change the fundamental architecture but can improve training efficiency.

## Model Structure

The GPT-2 model implemented in this project follows the architecture described in the original paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) by Radford et al. The key components of the model are:

1. **Token and Positional Embeddings**: Convert input tokens into embeddings and add positional information.

2. **Transformer Blocks**: A series of blocks, each containing:
   - **Multi-Head Attention**: Allows the model to attend to different parts of the input sequence.
   - **Layer Normalization**: Normalizes the outputs of the attention and feed-forward layers.
   - **Feed-Forward Neural Network**: Processes the attention output.

3. **Final Layer Normalization**: Applied after the last transformer block.

4. **Language Model Head**: A linear layer that projects the final hidden states to vocabulary-sized logits.

The model uses the following key classes:

- `GPT2`: The main model class that combines all components.
- `Block`: Represents a single transformer block.
- `CausalSelfAttention`: Implements the multi-head self-attention mechanism with causal masking.
- `MLP`: The feed-forward neural network used in each block.

## Project Structure

```
project/
├── config/
│   └── default_config.yaml
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data_loader.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gpt2-implementation.git
   cd gpt2-implementation
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:

```bash
python main.py --config config/default_config.yaml --mode train
```

This will start the training process using the settings specified in the config file. The script will log training progress and save model checkpoints periodically.

### Inference

To generate text using a trained model:

```bash
python main.py --config config/default_config.yaml --mode inference --prompt "Your prompt here"
```

Replace "Your prompt here" with the text you want to use as a starting point for generation.

## Configuration

The `config/default_config.yaml` file contains all the configurable parameters for the model and training process. You can modify this file to change:

- Model architecture (e.g., number of layers, embedding size)
- Training settings (e.g., batch size, learning rate)
- Data source
- Logging and checkpoint saving frequency

Here's an example of the configuration structure:

```yaml
model:
  block_size: 1024
  vocab_size: 50257
  n_layer: 12
  n_head: 12
  n_embd: 768

training:
  num_epochs: 50
  batch_size: 4
  sequence_length: 32
  learning_rate: 3e-4
  device: 'cuda'

data:
  input_file: 'input.txt'

logging:
  log_interval: 10
  save_interval: 1000
  model_save_path: 'checkpoints/'
```

## Author

**Yalala Mohit**

<div align="center">
  <p>If you find this project useful, please consider giving it a star!</p>
  <a href="https://github.com/yourusername/gpt2-implementation">
    <img src="https://img.shields.io/github/stars/yourusername/gpt2-implementation.svg?style=social&label=Star" alt="GitHub stars">
  </a>
</div>