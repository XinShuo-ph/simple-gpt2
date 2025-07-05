# Simple GPT-2 Transformer Implementation

A clean, educational implementation of a GPT-2 like transformer model in PyTorch.

## Features

- 🤖 **Complete GPT-2 Architecture**: Multi-head attention, feed-forward networks, layer normalization
- 🎯 **Causal Language Modeling**: Autoregressive text generation with causal masking
- 🔧 **Flexible Configuration**: Customizable model size, layers, heads, and vocabulary
- 📚 **Educational**: Well-commented code for learning transformer internals
- 🚀 **Ready to Use**: Includes training and inference scripts

## Model Architecture

The implementation includes:

- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Feed-Forward Networks**: Position-wise feed-forward layers with GELU activation
- **Layer Normalization**: Pre-norm architecture for stable training
- **Positional Embeddings**: Learned positional encodings
- **Causal Masking**: Autoregressive generation with causal attention masks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from gpt2_model import GPT2Model
import torch

# Create model
model = GPT2Model(
    vocab_size=50257,  # GPT-2 vocab size
    d_model=768,       # Hidden dimension
    num_heads=12,      # Number of attention heads
    num_layers=12,     # Number of transformer blocks
    max_seq_length=1024,
    d_ff=3072          # Feed-forward dimension
)

# Generate text
input_ids = torch.randint(0, 50257, (1, 10))
generated = model.generate(
    input_ids, 
    max_new_tokens=50, 
    temperature=0.8,
    top_k=50
)
```

## Model Configurations

### Small Model (Demo)
```python
model = GPT2Model(
    vocab_size=256,
    d_model=256,
    num_heads=8,
    num_layers=4,
    max_seq_length=128,
    d_ff=1024
)
# ~2M parameters
```

### GPT-2 Small
```python
model = GPT2Model(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_layers=12,
    max_seq_length=1024,
    d_ff=3072
)
# ~124M parameters
```

## Files

- `gpt2_model.py` - Main model implementation
- `requirements.txt` - Package dependencies
- `README.md` - This file

## Implementation Details

### Multi-Head Attention
- Scaled dot-product attention
- Causal masking for autoregressive generation
- Dropout for regularization

### Feed-Forward Networks
- Two linear layers with GELU activation
- Dropout for regularization

### Text Generation
- Top-k sampling
- Temperature scaling
- Causal mask enforcement

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy
- tqdm (for training progress)

## License

This project is provided for educational purposes. Feel free to use and modify.

## Acknowledgments

This implementation is inspired by:
- Original GPT-2 paper: "Language Models are Unsupervised Multitask Learners"
- Attention mechanism from "Attention Is All You Need"
- Various open-source transformer implementations