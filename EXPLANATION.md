# Complete Guide: Understanding the Language Model from Scratch

This guide explains every component of the language model implementation in beginner-friendly terms.

## Table of Contents
1. [What is a Language Model?](#what-is-a-language-model)
2. [Project Structure Overview](#project-structure-overview)
3. [Component 1: Tokenizer](#component-1-tokenizer)
4. [Component 2: Model Architecture](#component-2-model-architecture)
5. [Component 3: Training Process](#component-3-training-process)
6. [Component 4: Inference/Generation](#component-4-inferencegeneration)
7. [How Everything Works Together](#how-everything-works-together)

---

## What is a Language Model?

A **language model** is a machine learning model that learns to predict the next word (or character) in a sequence. Think of it like autocomplete on your phone - it suggests what you might type next based on what you've already typed.

**Example:**
- Input: "The quick brown"
- Model predicts: "fox" (because "The quick brown fox" is a common phrase)

Our model learns patterns from training data and uses them to generate new text.

---

## Project Structure Overview

```
MiniGPT/
├── tokenizer.py    # Converts text ↔ numbers
├── model.py        # The neural network architecture
├── train.py        # Training the model
├── inference.py    # Generating text
└── main.py         # Command-line interface
```

**The Flow:**
1. **Tokenizer**: Converts text to numbers (computers work with numbers!)
2. **Model**: A neural network that learns patterns
3. **Training**: Teaches the model using examples
4. **Inference**: Uses the trained model to generate new text

---

## Component 1: Tokenizer (`tokenizer.py`)

### What is Tokenization?

Computers can't understand words directly - they need numbers. **Tokenization** converts text into numbers (token IDs) that the model can process.

### How Our Tokenizer Works

#### 1. **Building Vocabulary**

```python
def build_vocab(self, texts: List[str], min_freq: int = 1):
```

- Reads all training text
- Counts how often each character appears
- Creates a dictionary mapping each character to a unique number
- Example: `{"a": 14, "b": 15, " ": 4, ...}`

**Special Tokens:**
- `<pad>` (0): Padding - used to make sequences the same length
- `<unk>` (1): Unknown - for characters not in vocabulary
- `<bos>` (2): Beginning of sequence
- `<eos>` (3): End of sequence

#### 2. **Encoding (Text → Numbers)**

```python
def encode(self, text: str) -> List[int]:
```

**Example:**
- Input: `"hello"`
- Output: `[2, 21, 18, 25, 25, 28, 3]`
  - `2` = `<bos>` (start)
  - `21, 18, 25, 25, 28` = "hello"
  - `3` = `<eos>` (end)

#### 3. **Decoding (Numbers → Text)**

```python
def decode(self, token_ids: List[int]) -> str:
```

**Example:**
- Input: `[21, 18, 25, 25, 28]`
- Output: `"hello"`

### Why Character-Level?

We use **character-level tokenization** (each character = one token) because:
- Simple to implement
- Works with any text
- No need for a pre-built vocabulary

**Trade-off**: More tokens per word (e.g., "hello" = 5 tokens), but simpler!

---

## Component 2: Model Architecture (`model.py`)

This is the "brain" of our language model - a **Transformer** neural network.

### Overview: What is a Transformer?

A Transformer is a type of neural network architecture that:
1. Processes sequences (like text)
2. Uses **attention** to understand relationships between words
3. Can learn long-range dependencies (words far apart can relate)

### Architecture Breakdown

```
Input Text
    ↓
Token Embeddings (convert tokens to vectors)
    ↓
Positional Encoding (add position information)
    ↓
Transformer Blocks (x4 layers)
    ├── Multi-Head Attention
    └── Feed-Forward Network
    ↓
Output Layer
    ↓
Predictions (probabilities for next token)
```

Let's break down each component:

---

### 2.1 Token Embeddings

**What it does:** Converts token IDs (numbers) into dense vectors (arrays of numbers)

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

**Why?** 
- Raw token IDs (like `14`) don't have meaning
- Embeddings (like `[0.2, -0.5, 0.8, ...]`) can capture relationships
- Similar tokens get similar embeddings

**Example:**
- Token `14` ("a") → `[0.2, -0.1, 0.5, ...]`
- Token `15` ("b") → `[0.3, -0.2, 0.4, ...]` (similar because they're both letters)

---

### 2.2 Positional Encoding

**Problem:** The model needs to know word order!
- "Dog bites man" ≠ "Man bites dog"

**Solution:** Add position information to embeddings

```python
self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
```

**How it works:**
- Uses sine and cosine functions to encode position
- Position 0 gets one pattern, position 1 gets another, etc.
- Added to token embeddings so the model knows "this token is at position 5"

**Visual:**
```
Token Embedding:  [0.2, -0.1, 0.5]
Position Encoding: [0.1, 0.3, -0.2]
                  ────────────────
Result:            [0.3, 0.2, 0.3]  (combined)
```

---

### 2.3 Multi-Head Attention

**This is the KEY innovation of Transformers!**

#### What is Attention?

**Attention** lets the model "look at" different parts of the input when making predictions.

**Example:**
- Input: "The cat sat on the mat"
- To predict "mat", the model should pay attention to:
  - "sat" (verb - what happened)
  - "on" (preposition - location)
  - "the" (article - noun coming)

#### How Multi-Head Attention Works

```python
class MultiHeadAttention(nn.Module):
```

**Step 1: Create Queries, Keys, Values**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I represent?"
- **Value (V)**: "What information do I contain?"

**Step 2: Calculate Attention Scores**
```
scores = Q × K^T / √d_k
```
- Measures how much each token should attend to others
- Higher score = more attention

**Step 3: Apply Softmax**
- Converts scores to probabilities (sum to 1)
- Ensures we "attend" to tokens proportionally

**Step 4: Weighted Sum**
```
output = attention_weights × V
```
- Combines information from all tokens
- Tokens with higher attention contribute more

**Why "Multi-Head"?**
- Multiple attention mechanisms run in parallel
- Each head learns different relationships
- Example: Head 1 learns syntax, Head 2 learns semantics

---

### 2.4 Feed-Forward Network

**What it does:** Processes the attention output through a simple neural network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        self.linear1 = nn.Linear(d_model, d_ff)      # Expand
        self.linear2 = nn.Linear(d_ff, d_model)      # Compress
```

**Architecture:**
```
Input (128 dims)
    ↓
Linear Layer (128 → 512 dims)  [Expand]
    ↓
ReLU Activation
    ↓
Linear Layer (512 → 128 dims)  [Compress]
    ↓
Output (128 dims)
```

**Why?**
- Adds non-linearity (ReLU)
- Allows the model to learn complex transformations
- Think of it as "thinking" about the attended information

---

### 2.5 Transformer Block

Combines attention + feed-forward with **residual connections** and **layer normalization**

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))
        # Feed-forward with residual
        x = x + self.feed_forward(self.norm2(x))
        return x
```

**Key Concepts:**

1. **Residual Connections** (`x = x + ...`)
   - Adds the input to the output
   - Helps with training (gradient flow)
   - Allows the model to "skip" layers if needed

2. **Layer Normalization** (`self.norm1(x)`)
   - Normalizes values to a standard range
   - Stabilizes training
   - Makes the model more robust

**Why Stack Multiple Blocks?**
- Each block refines the understanding
- Block 1: Basic patterns
- Block 2: More complex patterns
- Block 3: Even more complex
- Block 4: Final refinement

---

### 2.6 Output Layer

```python
self.head = nn.Linear(d_model, vocab_size)
```

**What it does:**
- Takes the final hidden state (128 dims)
- Projects to vocabulary size (40 dims in our case)
- Each dimension = probability of that token being next

**Example Output:**
```
[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, ...]
 ↑     ↑     ↑     ↑     ↑     ↑
pad   unk   bos   eos   ' '   '!'  ...
```

The model predicts token with ID 5 (probability 0.20) is most likely next!

---

## Component 3: Training Process (`train.py`)

### What is Training?

**Training** is teaching the model by showing it examples and correcting its mistakes.

**Analogy:** Like teaching a child to read:
1. Show examples: "The cat sat on the mat"
2. Ask: "What comes after 'sat on the'?"
3. If wrong, correct: "It's 'mat'"
4. Repeat many times

### Training Steps

#### Step 1: Prepare Data

```python
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a great programming language.",
    ...
]
```

- Each sentence is a training example
- The model learns patterns from these

#### Step 2: Create Dataset

```python
class TextDataset(Dataset):
    def __getitem__(self, idx):
        # Encode text to numbers
        token_ids = tokenizer.encode(text)
        
        # Create input and target
        input_ids = token_ids[:-1]   # All except last
        labels = token_ids[1:]       # All except first
```

**Example:**
- Text: "hello"
- Encoded: `[2, 21, 18, 25, 25, 28, 3]`
- Input:  `[2, 21, 18, 25, 25, 28]`  (predict next)
- Target: `[21, 18, 25, 25, 28, 3]`  (what should be next)

**Why?** The model learns: "Given [2, 21, 18], predict 25"

#### Step 3: Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass: Get predictions
        logits = model(input_ids)
        
        # Calculate loss: How wrong are we?
        loss = criterion(logits, labels)
        
        # Backward pass: Update weights
        loss.backward()
        optimizer.step()
```

**Detailed Breakdown:**

1. **Forward Pass**
   - Input tokens go through the model
   - Model outputs predictions (logits)
   - Example: Model predicts token 5 with probability 0.2

2. **Calculate Loss**
   - Compare predictions to actual next token
   - Use Cross-Entropy Loss
   - Lower loss = better predictions

3. **Backward Pass (Backpropagation)**
   - Calculate gradients (how to adjust weights)
   - Update model weights to reduce loss
   - Model gets slightly better!

4. **Repeat**
   - Process all batches
   - Complete one epoch
   - Start next epoch

#### Step 4: Loss Function

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
```

**What it does:**
- Measures how far predictions are from truth
- Lower = better
- Ignores padding tokens (they're not real predictions)

**Example:**
- True next token: `25` ("o")
- Model prediction: `[0.1, 0.05, 0.2, 0.15, 0.5, ...]`
  - High probability (0.5) at position 25 ✓ Good!
- If model predicted: `[0.5, 0.1, 0.2, ...]`
  - High probability at wrong position ✗ Bad! High loss

#### Step 5: Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**What it does:**
- Updates model weights based on gradients
- **Learning Rate**: How big steps to take
  - Too high: Overshoots, unstable
  - Too low: Too slow, might not learn

**AdamW**: An advanced optimizer that adapts learning rate per parameter

#### Step 6: Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Why?** Prevents gradients from becoming too large (exploding gradients)
- Clips gradients to max norm of 1.0
- Stabilizes training

---

## Component 4: Inference/Generation (`inference.py`)

### What is Inference?

**Inference** is using a trained model to generate new text.

**Key Difference:**
- **Training**: Model learns from examples
- **Inference**: Model generates new text

### How Text Generation Works

#### Step 1: Encode Prompt

```python
prompt = "hello"
input_ids = tokenizer.encode(prompt)  # [21, 18, 25, 25, 28]
```

#### Step 2: Generate Autoregressively

**Autoregressive** = generating one token at a time, using previous tokens

```python
for _ in range(max_new_tokens):
    # Get predictions
    logits = model(input_ids)
    
    # Get next token probabilities
    next_token_logits = logits[:, -1, :]  # Last position
    
    # Sample a token
    next_token = sample(next_token_logits)
    
    # Add to sequence
    input_ids = torch.cat([input_ids, next_token])
```

**Visual Example:**
```
Step 1: "hello" → predict " " (space)
Step 2: "hello " → predict "w"
Step 3: "hello w" → predict "o"
Step 4: "hello wo" → predict "r"
Step 5: "hello wor" → predict "l"
Step 6: "hello worl" → predict "d"
Result: "hello world"
```

#### Step 3: Sampling Strategies

**Problem:** Model outputs probabilities, not a single token. How do we choose?

**Option 1: Greedy (Always pick highest)**
```python
next_token = torch.argmax(probs)  # Always pick most likely
```
- **Pro:** Deterministic, fast
- **Con:** Can be repetitive, boring

**Option 2: Random Sampling**
```python
next_token = torch.multinomial(probs, 1)  # Sample based on probabilities
```
- **Pro:** More diverse, creative
- **Con:** Can be too random, incoherent

**Option 3: Temperature Sampling**
```python
logits = logits / temperature
probs = softmax(logits)
next_token = torch.multinomial(probs, 1)
```

**Temperature Effect:**
- `temperature = 1.0`: Normal randomness
- `temperature < 1.0`: More conservative (picks likely tokens)
- `temperature > 1.0`: More creative (picks less likely tokens)

**Option 4: Top-k Sampling**
```python
top_k_logits, top_k_indices = torch.topk(logits, k=50)
# Only sample from top 50 most likely tokens
```

**Why?** Prevents sampling very unlikely tokens (reduces gibberish)

**Our Implementation Uses:**
- Temperature sampling (default 0.8)
- Top-k sampling (default 50)
- Combined for balanced creativity/coherence

#### Step 4: Early Stopping

```python
if next_token == eos_token_id:
    break  # Stop generating
```

- Stops when model generates `<eos>` (end-of-sequence)
- Prevents infinite generation

#### Step 5: Decode Output

```python
generated_text = tokenizer.decode(generated_ids)
```

- Converts token IDs back to text
- Removes special tokens
- Returns readable text!

---

## How Everything Works Together

### Complete Flow: Training → Inference

#### Phase 1: Training

```
1. Load training data
   ↓
2. Build tokenizer vocabulary
   ↓
3. Create model (random weights)
   ↓
4. For each epoch:
   a. For each batch:
      - Encode text → token IDs
      - Forward pass → predictions
      - Calculate loss
      - Backward pass → update weights
   ↓
5. Save model + tokenizer
```

#### Phase 2: Inference

```
1. Load trained model + tokenizer
   ↓
2. User provides prompt: "hello"
   ↓
3. Encode prompt: [21, 18, 25, 25, 28]
   ↓
4. Generate tokens autoregressively:
   - Model predicts next token
   - Sample using temperature + top-k
   - Append to sequence
   - Repeat until max_tokens or EOS
   ↓
5. Decode generated tokens → text
   ↓
6. Return: "hello world"
```

### Example: Complete Generation

**Input:** "The quick brown"

**Step-by-step:**
1. Encode: `[13, 21, 18, 4, 30, 34, 22, 16, 24, 4, 15, 31, 28, 36, 27]`
2. Model processes through transformer blocks
3. Output layer predicts: `"fox"` (token 19) with highest probability
4. Sample: `19` → "f"
5. Append: `[..., 19]`
6. Model processes again, predicts: `28` → "o"
7. Continue: `"fox jumps over..."`

**Final Output:** "The quick brown fox jumps over the lazy dog."

---

## Key Concepts Summary

### 1. **Embeddings**
- Convert discrete tokens to continuous vectors
- Capture semantic relationships

### 2. **Attention**
- Model "looks at" relevant parts of input
- Learns relationships between tokens

### 3. **Residual Connections**
- Helps information flow through deep networks
- Enables training of very deep models

### 4. **Autoregressive Generation**
- Generate one token at a time
- Each token depends on all previous tokens

### 5. **Sampling**
- Balance between creativity and coherence
- Temperature and top-k control randomness

---

## Why This Architecture Works

1. **Attention**: Understands context and relationships
2. **Multiple Layers**: Learns hierarchical patterns
3. **Residual Connections**: Enables deep networks
4. **Positional Encoding**: Preserves word order
5. **Autoregressive**: Natural for language (left-to-right)

---

## Limitations & Improvements

### Current Limitations:
- Small model (128 dims, 4 layers)
- Character-level tokenization (inefficient)
- Small training data
- No validation set

### Possible Improvements:
1. **Larger Model**: More parameters = better learning
2. **Subword Tokenization**: BPE, WordPiece (more efficient)
3. **More Data**: Larger, diverse dataset
4. **Learning Rate Scheduling**: Adjust LR during training
5. **Validation**: Monitor overfitting
6. **Better Optimizers**: AdamW with warmup
7. **Regularization**: Dropout, weight decay

---

## Next Steps for Learning

1. **Experiment with Hyperparameters**
   - Try different learning rates
   - Adjust model size
   - Change number of layers

2. **Add More Training Data**
   - Use a larger text file
   - Try different domains (code, stories, etc.)

3. **Implement Improvements**
   - Add validation loop
   - Implement learning rate scheduling
   - Try different architectures

4. **Study Advanced Topics**
   - BPE tokenization
   - Pre-trained models (GPT-2, GPT-3)
   - Fine-tuning
   - Prompt engineering

---

## Conclusion

You now understand:
- ✅ How tokenization works
- ✅ Transformer architecture components
- ✅ Training process (forward/backward pass)
- ✅ Text generation (autoregressive sampling)
- ✅ How everything connects

This is a **complete, working language model** from scratch! While it's minimal, it demonstrates all core concepts used in models like GPT, BERT, and others.

Happy learning! 🚀

