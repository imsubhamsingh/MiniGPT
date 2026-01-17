"""
Main entry point - provides a simple interface to the language model.
"""

import argparse
import os
import torch

from model import MiniGPT
from tokenizer import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser(description="Minimal Language Model")
    parser.add_argument(
        "mode",
        choices=["train", "infer", "interactive"],
        help="Mode: train, infer, or interactive"
    )
    parser.add_argument("--model_dir", type=str, default="./model_output", help="Model directory")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Import and run training
        from train import main as train_main
        import sys
        # Build sys.argv with all training arguments
        train_args = [
            "train.py",
            "--output_dir", args.model_dir,
            "--device", args.device,
            "--num_epochs", str(args.num_epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--max_length", str(args.max_length),
            "--d_model", str(args.d_model),
            "--n_heads", str(args.n_heads),
            "--n_layers", str(args.n_layers),
        ]
        sys.argv = train_args
        train_main()
    
    elif args.mode == "infer":
        if not args.prompt:
            print("Error: --prompt is required for inference mode")
            return
        
        if not os.path.exists(args.model_dir):
            print(f"Error: Model directory '{args.model_dir}' not found!")
            print("Please train a model first using: python main.py train")
            return
        
        from inference import generate_text
        
        print("Loading model...")
        model = MiniGPT.from_pretrained(args.model_dir)
        tokenizer = SimpleTokenizer.from_pretrained(args.model_dir)
        print("Model loaded!")
        
        device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            device=device,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}")
    
    elif args.mode == "interactive":
        if not os.path.exists(args.model_dir):
            print(f"Error: Model directory '{args.model_dir}' not found!")
            print("Please train a model first using: python main.py train")
            return
        
        from inference import interactive_mode
        
        device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        interactive_mode(args.model_dir, device)


if __name__ == "__main__":
    main()
