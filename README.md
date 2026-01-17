# MiniGPT: Minimal Language Model from Scratch

A minimal transformer-based language model implementation from scratch, similar to GPT. This project demonstrates how to build, train, and deploy a small language model using PyTorch, structured like models on Hugging Face.

## Documentation

📚 **Learning Resources:**

- **[Complete Guide](EXPLANATION.md)** - Detailed beginner-friendly explanation of all components, how they work, and the complete architecture
- **[ChatGPT Comparison](CHATGPT_COMPARISON.md)** - Understand how this model compares to ChatGPT and what makes large language models powerful

## Features

- **Transformer Architecture**: GPT-like decoder-only transformer with:
  - Multi-head self-attention
  - Position-wise feed-forward networks
  - Layer normalization and residual connections
  - Positional encoding

- **Complete Training Pipeline**: 
  - Data preparation and tokenization
  - Training loop with checkpointing
  - Model saving/loading

- **Inference**: 
  - Text generation with temperature and top-k sampling
  - Interactive chat mode
  - Batch inference

- **Hugging Face-like Structure**:
  - Model configuration (config.json)
  - Tokenizer vocabulary (vocab.json)
  - Model weights (pytorch_model.bin)
  - Tokenizer configuration

## Project Structure

```
MiniGPT/
├── model.py          # Transformer model architecture
├── tokenizer.py      # Simple character-level tokenizer
├── train.py          # Training script
├── inference.py      # Inference and generation script
├── main.py           # Main entry point
├── pyproject.toml    # Dependencies
└── README.md         # This file
```

## Installation

1. Install dependencies:
```bash
uv sync
```

Or manually:
```bash
pip install torch transformers tokenizers numpy tqdm
```

## Quick Start

### 1. Train the Model

Train a small language model on sample data:

```bash
python main.py train --model_dir ./model_output
```

Or use the training script directly with more options:

```bash
python train.py \
    --output_dir ./model_output \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 4 \
    --device cpu
```

**Model Architecture Parameters:**
- `d_model`: Model dimension (default: 128)
- `n_heads`: Number of attention heads (default: 4)
- `n_layers`: Number of transformer layers (default: 4)
- `d_ff`: Feed-forward dimension (default: d_model * 4)

### 2. Generate Text

**Interactive Mode:**
```bash
python main.py interactive --model_dir ./model_output
```

**Single Generation:**
```bash
python main.py infer --model_dir ./model_output --prompt "The quick brown"
```

Or use the inference script directly:
```bash
python inference.py \
    --model_dir ./model_output \
    --prompt "Hello world" \
    --max_new_tokens 50 \
    --temperature 0.8 \
    --top_k 50
```

## Model Files

After training, the model directory will contain:

```
model_output/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── vocab.json               # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer configuration
└── checkpoints/             # Training checkpoints
    └── checkpoint_epoch_*.pt
```

## Usage Examples

### Programmatic Usage

```python
from model import MiniGPT
from tokenizer import SimpleTokenizer

# Load model and tokenizer
model = MiniGPT.from_pretrained("./model_output")
tokenizer = SimpleTokenizer.from_pretrained("./model_output")

# Encode text
text = "Hello world"
input_ids = tokenizer.encode(text, add_special_tokens=False)
input_ids = torch.tensor([input_ids])

# Generate
model.eval()
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )

# Decode
generated_text = tokenizer.decode(generated_ids[0].tolist())
print(generated_text)
```

### Training on Custom Data

Modify `train.py` to use your own dataset:

```python
def prepare_sample_data():
    # Replace with your data
    return [
        "Your text data here...",
        "More text...",
        # ...
    ]
```

Or load from a file:

```python
def prepare_sample_data():
    with open("your_data.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]
```

## Model Architecture

The model is a minimal GPT-like transformer:

1. **Token Embeddings**: Maps tokens to dense vectors
2. **Positional Encoding**: Adds positional information
3. **Transformer Blocks**: Stack of:
   - Multi-head self-attention
   - Feed-forward network
   - Layer normalization
   - Residual connections
4. **Output Head**: Projects to vocabulary size

**Default Configuration:**
- Vocabulary: Character-level (built from training data)
- Model dimension: 128
- Attention heads: 4
- Layers: 4
- Feed-forward: 512
- Max sequence length: 256

## Training Details

- **Loss Function**: Cross-entropy (ignoring padding tokens)
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3 (default)
- **Gradient Clipping**: 1.0
- **Batch Size**: 4 (default, adjust based on memory)

## Limitations

This is a **minimal educational implementation**:

- Very small model size (suitable for learning, not production)
- Character-level tokenization (not subword/word-level)
- Trained on small sample data
- No advanced features like:
  - Mixed precision training
  - Distributed training
  - Advanced optimizers
  - Learning rate scheduling

For production use, consider:
- Using Hugging Face Transformers library
- Pre-trained models (GPT-2, GPT-3, etc.)
- Proper tokenizers (BPE, SentencePiece)
- Larger models and datasets

## Extending the Model

You can extend this model by:

1. **Adding more layers/parameters** for better performance
2. **Implementing subword tokenization** (BPE, WordPiece)
3. **Adding learning rate scheduling**
4. **Implementing validation loop**
5. **Adding more advanced sampling strategies**
6. **Supporting batch generation**

## Similar to Hugging Face Models

This implementation follows Hugging Face conventions:

- `config.json`: Model configuration
- `pytorch_model.bin`: Model weights
- `vocab.json`: Tokenizer vocabulary
- `tokenizer_config.json`: Tokenizer settings
- `from_pretrained()` / `save_pretrained()` methods

You can use this as a foundation to understand how models work before using larger pre-trained models.

## Requirements

- Python 3.13+
- PyTorch 2.0+
- NumPy
- tqdm (for progress bars)

## License

This is an educational project for learning purposes.
