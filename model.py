"""
Minimal GPT-like Transformer Language Model from Scratch

This file contains the neural network architecture for our language model.
Think of it as the "brain" that learns to predict the next character in text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This is the KEY component that makes transformers powerful!
    
    What it does: Allows the model to "look at" and understand relationships 
    between different parts of the input text. Like when you read a sentence,
    you pay attention to different words to understand the meaning.
    
    "Multi-head" means we do this attention process multiple times in parallel,
    each time learning different types of relationships (like grammar vs meaning).
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize the attention mechanism.
        
        Args:
            d_model: The size of our model's hidden states (like 128)
                    Think of it as how many numbers we use to represent each token
            n_heads: Number of attention "heads" (like 4)
                    Each head learns different relationships, so more heads = more understanding
            dropout: Probability of randomly turning off neurons during training (0.1 = 10%)
                    This helps prevent overfitting (memorizing instead of learning)
        """
        # Initialize the parent PyTorch Module class
        super().__init__()
        
        # Make sure d_model can be evenly divided by n_heads
        # Example: if d_model=128 and n_heads=4, then each head gets 128/4=32 dimensions
        assert d_model % n_heads == 0
        
        # Store these values so we can use them later
        self.d_model = d_model  # Total model dimension
        self.n_heads = n_heads  # Number of attention heads
        self.d_k = d_model // n_heads  # Dimension per head (128/4 = 32)
        
        # Create linear transformation layers (like y = Wx + b)
        # These convert our input into Query, Key, and Value vectors
        # Think of it like asking questions (Q), providing answers (K), and getting information (V)
        self.w_q = nn.Linear(d_model, d_model)  # Query transformation
        self.w_k = nn.Linear(d_model, d_model)  # Key transformation
        self.w_v = nn.Linear(d_model, d_model)  # Value transformation
        self.w_o = nn.Linear(d_model, d_model)  # Output transformation (combines all heads)
        
        # Dropout layer: randomly sets some values to 0 during training
        # This helps the model generalize better (not memorize training data)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass: process the input through attention.
        
        Args:
            x: Input tensor [batch_size, sequence_length, d_model]
               Example: [2, 10, 128] means 2 sentences, 10 tokens each, 128 dimensions per token
            mask: Optional mask to hide certain tokens (like padding or future tokens)
        
        Returns:
            Output tensor with the same shape as input, but with updated representations
        """
        # Get the dimensions of our input
        batch_size, seq_len, d_model = x.size()
        # Example: if x is [2, 10, 128], then batch_size=2, seq_len=10, d_model=128
        
        # Step 1: Create Query, Key, and Value vectors
        # We transform the input x into Q, K, V using linear layers
        # Then reshape to separate into multiple heads
        # Example: [2, 10, 128] -> [2, 4, 10, 32] (2 batches, 4 heads, 10 tokens, 32 dims per head)
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Step 2: Calculate attention scores
        # This measures how much each token should "pay attention" to every other token
        # We multiply Q and K^T (transpose), then divide by sqrt(d_k) to keep values stable
        # Higher scores = more attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Step 3: Apply mask (if provided)
        # Masks hide certain tokens (set their attention to very negative number)
        # When we apply softmax, these become 0 (no attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # -1e9 is like negative infinity
        
        # Step 4: Convert scores to probabilities using softmax
        # Softmax makes all scores sum to 1, so they become probabilities
        # Example: [0.1, 0.3, 0.6] means 10% attention to token 1, 30% to token 2, 60% to token 3
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights (randomly zero some out during training)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Apply attention weights to Value vectors
        # This is like taking a weighted average: tokens with higher attention contribute more
        context = torch.matmul(attention_weights, V)
        
        # Step 6: Combine all heads back together
        # We transpose and reshape to combine the 4 heads back into one representation
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Step 7: Final output transformation
        # Apply one more linear layer to get the final output
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    This is a simple 2-layer neural network that processes each token independently.
    Think of it as "thinking" about the information from attention.
    
    It expands the dimensions (makes it bigger), applies ReLU (adds non-linearity),
    then compresses back to original size.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model: Input/output dimension (e.g., 128)
            d_ff: Hidden layer dimension (usually 4x d_model, e.g., 512)
                  Bigger = more capacity to learn complex patterns
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # First linear layer: expands from d_model to d_ff
        # Example: 128 -> 512 (makes it bigger)
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: compresses back from d_ff to d_model
        # Example: 512 -> 128 (makes it smaller again)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the feed-forward network.
        
        Process: Expand -> ReLU (activation) -> Dropout -> Compress
        
        ReLU (Rectified Linear Unit) = max(0, x)
        It adds non-linearity, allowing the model to learn complex patterns.
        Without it, the model would just be linear transformations (limited power).
        """
        # Expand: [batch, seq_len, 128] -> [batch, seq_len, 512]
        expanded = self.linear1(x)
        
        # Apply ReLU activation (makes negative values 0, keeps positive values)
        activated = F.relu(expanded)
        
        # Apply dropout (randomly zero some values during training)
        dropped = self.dropout(activated)
        
        # Compress back: [batch, seq_len, 512] -> [batch, seq_len, 128]
        output = self.linear2(dropped)
        
        return output


class TransformerBlock(nn.Module):
    """
    A complete transformer block (one layer of the model).
    
    This combines attention and feed-forward with some important tricks:
    - Residual connections: helps information flow through deep networks
    - Layer normalization: stabilizes training
    
    We stack multiple of these blocks to build a deep network.
    Each block refines the understanding of the input.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize one transformer block.
        
        Args:
            d_model: Model dimension (e.g., 128)
            n_heads: Number of attention heads (e.g., 4)
            d_ff: Feed-forward hidden dimension (e.g., 512)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Create the attention mechanism
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Create the feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization: normalizes values to have mean=0, std=1
        # This helps training be more stable
        self.norm1 = nn.LayerNorm(d_model)  # Before attention
        self.norm2 = nn.LayerNorm(d_model)  # Before feed-forward
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass through one transformer block.
        
        The flow:
        1. Normalize input
        2. Apply attention
        3. Add residual connection (x = x + attention_output)
        4. Normalize again
        5. Apply feed-forward
        6. Add residual connection again
        
        Residual connections (x = x + ...) are crucial!
        They allow gradients to flow through deep networks,
        making it possible to train very deep models.
        """
        # Part 1: Self-attention with residual connection
        # Normalize first (helps with training stability)
        normalized = self.norm1(x)
        
        # Apply attention
        attention_output = self.attention(normalized, mask)
        
        # Add residual connection: x = x + attention_output
        # This is like saying "keep the original, but add the improvements"
        # It helps information flow through deep networks
        x = x + self.dropout(attention_output)
        
        # Part 2: Feed-forward with residual connection
        # Normalize again
        normalized = self.norm2(x)
        
        # Apply feed-forward network
        ff_output = self.feed_forward(normalized)
        
        # Add residual connection again
        x = x + self.dropout(ff_output)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding: tells the model where each token is in the sequence.
    
    Problem: The model needs to know word order!
    "Dog bites man" is different from "Man bites dog"
    
    Solution: Add position information to each token's embedding.
    We use sine and cosine functions to encode position in a way the model can learn.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (must match token embedding dimension)
            max_len: Maximum sequence length we can handle (e.g., 5000 tokens)
        """
        super().__init__()
        
        # Create a matrix to store positional encodings
        # Shape: [max_len, d_model]
        # Each row is the encoding for that position (0, 1, 2, ...)
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices: [0, 1, 2, 3, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the divisor term for sine/cosine
        # This creates different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        # This allows broadcasting when we add it to token embeddings
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter, so it won't be trained)
        # But it will be saved/loaded with the model
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor):
        """
        Add positional encoding to input.
        
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
        
        Returns:
            x + positional_encoding (same shape as x)
        """
        # Add positional encoding to token embeddings
        # We only take the first seq_len positions from our pre-computed encoding
        # Example: if x has 10 tokens, we take positions 0-9 from pe
        return x + self.pe[:, :x.size(1)]


class MiniGPT(nn.Module):
    """
    The complete language model!
    
    This is like a mini version of GPT (Generative Pre-trained Transformer).
    
    Architecture flow:
    1. Convert token IDs to embeddings (numbers -> vectors)
    2. Add positional encoding (where is each token?)
    3. Process through multiple transformer blocks (understand relationships)
    4. Output probabilities for next token
    
    This model learns to predict the next character/token in a sequence.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        Initialize the language model.
        
        Args:
            vocab_size: Number of unique tokens in vocabulary (e.g., 40 characters)
            d_model: Model dimension - how many numbers represent each token (e.g., 128)
                    Bigger = more capacity, but slower and needs more memory
            n_heads: Number of attention heads (e.g., 4)
                    More heads = can learn more types of relationships
            n_layers: Number of transformer blocks to stack (e.g., 4)
                     More layers = deeper understanding, but harder to train
            d_ff: Feed-forward hidden dimension (usually 4x d_model, e.g., 512)
            max_seq_len: Maximum sequence length we can handle (e.g., 256 tokens)
            dropout: Dropout probability for regularization (0.1 = 10%)
            pad_token_id: ID of padding token (usually 0)
                         Padding is used to make all sequences the same length
        """
        super().__init__()
        
        # Store all the configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Step 1: Token Embedding Layer
        # Converts token IDs (like 14, 15, 16) to dense vectors (like [0.2, -0.1, 0.5, ...])
        # vocab_size = number of tokens, d_model = size of each embedding vector
        # padding_idx tells it which token to use for padding (usually 0)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Step 2: Positional Encoding
        # Adds position information so model knows word order
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Step 3: Stack of Transformer Blocks
        # We create n_layers transformer blocks and put them in a list
        # Each block refines the understanding of the input
        # Block 1: Basic patterns
        # Block 2: More complex patterns
        # Block 3: Even more complex
        # Block 4: Final refinement
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Step 4: Output Layer
        # Final layer normalization (stabilizes the output)
        self.ln_f = nn.LayerNorm(d_model)
        
        # Final linear layer: converts from d_model to vocab_size
        # This gives us probabilities for each token in vocabulary
        # Example: [128 dims] -> [40 dims] (one probability per token)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize all weights with small random values
        # This is important - random initialization helps the model learn
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize weights for linear layers and embeddings.
        
        We use small random values (mean=0, std=0.02) because:
        - Too large = training becomes unstable
        - Too small = model learns too slowly
        - Just right = model learns efficiently
        
        This is called "Xavier/Glorot initialization" style.
        """
        # For linear layers (fully connected layers)
        if isinstance(module, nn.Linear):
            # Initialize weights with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Initialize biases to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # For embedding layers
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass: process input through the entire model.
        
        This is what happens when we give the model some text and ask:
        "What token should come next?"
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
                      Example: [[14, 15, 16]] means one sentence with tokens 14, 15, 16
            attention_mask: Optional mask to ignore padding tokens
                           [batch_size, seq_len] where 1 = real token, 0 = padding
        
        Returns:
            logits: Predictions [batch_size, seq_len, vocab_size]
                   For each position, gives probability for each possible next token
                   Example: [[[0.01, 0.05, 0.10, ...], ...]]
                            This means at position 0, token 0 has prob 0.01, token 1 has 0.05, etc.
        """
        batch_size, seq_len = input_ids.size()
        
        # Create attention mask if not provided
        # 1 = pay attention to this token, 0 = ignore (padding)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Create causal mask: prevents model from "cheating" by looking at future tokens
        # This is a lower triangular matrix:
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        # Token 0 can only see token 0
        # Token 1 can see tokens 0 and 1
        # Token 2 can see tokens 0, 1, and 2
        # This is crucial for language modeling (predicting next token)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Combine attention mask with causal mask
        # Both must be 1 for attention to be allowed
        mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
        
        # Step 1: Convert token IDs to embeddings
        # [batch, seq_len] -> [batch, seq_len, d_model]
        # Example: [[14, 15]] -> [[[0.2, -0.1, ...], [0.3, 0.1, ...]]]
        x = self.token_embedding(input_ids)
        
        # Step 2: Add positional encoding
        # Tells model where each token is in the sequence
        x = self.positional_encoding(x)
        
        # Step 3: Apply dropout (only during training)
        x = self.dropout(x)
        
        # Step 4: Process through all transformer blocks
        # Each block refines the understanding
        for block in self.transformer_blocks:
            x = block(x, mask.squeeze(1))  # squeeze removes extra dimensions from mask
        
        # Step 5: Final layer normalization
        x = self.ln_f(x)
        
        # Step 6: Project to vocabulary size
        # [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
        # Now we have probabilities for each possible next token
        logits = self.head(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        pad_token_id: int = 0,
        eos_token_id: int = 3
    ):
        """
        Generate new text autoregressively (one token at a time).
        
        How it works:
        1. Start with a prompt (like "hello")
        2. Predict next token
        3. Add it to the sequence
        4. Repeat until we have enough tokens or hit end token
        
        Args:
            input_ids: Starting text as token IDs [batch_size, seq_len]
                      Example: [[21, 18, 25, 25, 28]] = "hello"
            max_new_tokens: Maximum number of tokens to generate (e.g., 50)
            temperature: Controls randomness (0.0-2.0)
                         Lower (0.5) = more conservative, picks likely tokens
                         Higher (1.5) = more creative, picks less likely tokens
                         Default (1.0) = balanced
            top_k: Only consider top K most likely tokens (e.g., 50)
                   None = consider all tokens
                   This prevents generating very unlikely tokens (reduces gibberish)
            pad_token_id: ID of padding token (usually 0)
            eos_token_id: ID of end-of-sequence token (usually 3)
                         When we generate this, we stop
        
        Returns:
            Generated token IDs including the original prompt
            [batch_size, original_seq_len + generated_tokens]
        """
        # Set model to evaluation mode (turns off dropout, batch norm updates, etc.)
        self.eval()
        
        # Don't compute gradients (we're not training, just generating)
        with torch.no_grad():
            # Generate tokens one at a time
            for _ in range(max_new_tokens):
                # Step 1: Get predictions from model
                # This gives us probabilities for each possible next token
                logits = self.forward(input_ids)
                
                # Step 2: Get predictions for the LAST token only
                # We only care about what comes after the current sequence
                # logits[:, -1, :] means: for all batches, last position, all vocab tokens
                next_token_logits = logits[:, -1, :] / temperature
                
                # Step 3: Prevent generating padding tokens
                # Set padding token probability to negative infinity
                # When we apply softmax, this becomes 0 (impossible)
                next_token_logits[:, pad_token_id] = float('-inf')
                
                # Step 4: Top-k sampling (optional)
                # Only consider the top K most likely tokens
                # This prevents the model from picking very unlikely tokens
                if top_k is not None:
                    # Make sure k doesn't exceed vocabulary size
                    k = min(top_k, next_token_logits.size(-1))
                    if k > 0:
                        # Get top k tokens and their scores
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                        
                        # Set all other tokens to negative infinity
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        
                        # Put back only the top k tokens
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Step 5: Convert logits to probabilities using softmax
                # Softmax makes all probabilities sum to 1.0
                # Example: [0.1, 0.3, 0.6] means 10% chance token 0, 30% token 1, 60% token 2
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Step 6: Sample one token based on probabilities
                # This randomly picks a token, but more likely tokens are picked more often
                # Example: if token 2 has 60% probability, it will be picked 60% of the time
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Step 7: Check if we generated end-of-sequence token
                # If yes, stop generating (we're done!)
                if next_token.item() == eos_token_id:
                    break
                
                # Step 8: Add the new token to our sequence
                # Append it to input_ids so it becomes part of the context for next prediction
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Step 9: Safety check - stop if sequence gets too long
                if input_ids.size(1) >= self.max_seq_len:
                    break
        
        # Return the complete sequence (original prompt + generated tokens)
        return input_ids
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model to disk.
        
        This saves:
        1. Model weights (all the learned parameters)
        2. Configuration (model architecture settings)
        
        Args:
            save_directory: Where to save the model (e.g., "./model_output")
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights (all the learned parameters)
        # This is the "brain" - all the numbers the model learned during training
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration (model architecture settings)
        # This tells us how to rebuild the model structure
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout.p,  # Get dropout probability
            "pad_token_id": self.pad_token_id,
        }
        
        # Write config to JSON file
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """
        Load a saved model from disk.
        
        This is the opposite of save_pretrained:
        1. Read the configuration
        2. Create a new model with that configuration
        3. Load the saved weights into it
        
        Args:
            model_directory: Where the model is saved (e.g., "./model_output")
        
        Returns:
            A model with loaded weights, ready to use
        """
        import os
        
        # Step 1: Load configuration
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Step 2: Create a new model with the saved configuration
        # cls(**config) means: create MiniGPT with all the settings from config
        model = cls(**config)
        
        # Step 3: Load the saved weights
        # This puts all the learned parameters back into the model
        model_path = os.path.join(model_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        return model
