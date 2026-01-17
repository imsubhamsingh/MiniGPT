"""
Training script for the minimal language model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm

from model import MiniGPT
from tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts: list, tokenizer: SimpleTokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def prepare_sample_data():
    """Prepare sample training data."""
    return [
        "Hello world"
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "Machine learning is fascinating.",
        "Hello world! This is a simple example.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns.",
        "Transformers are powerful neural network architectures.",
        "Artificial intelligence is transforming technology.",
        "Neural networks are inspired by the human brain.",
        "Training a model requires data and computation.",
    ] * 10  # Repeat to have more training examples


def train(
    model: MiniGPT,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_dir: str = "./checkpoints"
):
    """Train the model."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train minimal language model")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Minimal Language Model")
    print("=" * 60)
    
    # Prepare data
    print("\n1. Preparing data...")
    texts = prepare_sample_data()
    print(f"   Loaded {len(texts)} training examples")
    
    # Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, min_freq=1)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    print("\n3. Creating dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"   Dataset size: {len(dataset)}")
    
    # Create model
    print("\n4. Initializing model...")
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Model architecture:")
    print(f"     - Vocab size: {model.vocab_size}")
    print(f"     - Model dim: {model.d_model}")
    print(f"     - Attention heads: {model.n_heads}")
    print(f"     - Layers: {model.n_layers}")
    
    # Train
    print("\n5. Starting training...")
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"   Using device: {device}")
    
    train(
        model=model,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=os.path.join(args.output_dir, "checkpoints"),
    )
    
    # Save final model
    print("\n6. Saving model...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"   Model saved to: {args.output_dir}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

