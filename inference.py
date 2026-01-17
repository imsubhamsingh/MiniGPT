"""
Inference script for the minimal language model.
"""

import torch
import argparse
import os

from model import MiniGPT
from tokenizer import SimpleTokenizer


def generate_text(
    model: MiniGPT,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
):
    """Generate text from a prompt."""
    model.eval()
    model.to(device)
    
    # Adjust top_k to not exceed vocab size
    top_k = min(top_k, model.vocab_size) if top_k is not None else None
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    prompt_length = len(prompt_ids)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the newly generated tokens (exclude the original prompt)
    all_ids = generated_ids[0].cpu().tolist()
    generated_ids_only = all_ids[prompt_length:]
    
    # Decode only the generated part
    generated_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model_dir: str, device: str = "cpu"):
    """Interactive text generation."""
    print("=" * 60)
    print("Minimal Language Model - Interactive Mode")
    print("=" * 60)
    print("Type your prompt and press Enter to generate text.")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 60)
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    model = MiniGPT.from_pretrained(model_dir)
    tokenizer = SimpleTokenizer.from_pretrained(model_dir)
    print("Model loaded!")
    print()
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            print("Generating...", end=" ", flush=True)
            # Use vocab_size or 50, whichever is smaller
            top_k_value = min(50, model.vocab_size)
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_k=top_k_value,
                device=device,
            )
            print()
            print(f"Generated: {generated}")
            print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with minimal language model")
    parser.add_argument("--model_dir", type=str, default="./model_output", help="Model directory")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (if not provided, interactive mode)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found!")
        print("Please train a model first using train.py")
        return
    
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    if args.prompt:
        # Single generation
        print("Loading model...")
        model = MiniGPT.from_pretrained(args.model_dir)
        tokenizer = SimpleTokenizer.from_pretrained(args.model_dir)
        print("Model loaded!")
        print()
        
        print(f"Prompt: {args.prompt}")
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"Generated: {generated}")
    else:
        # Interactive mode
        interactive_mode(args.model_dir, device)


if __name__ == "__main__":
    main()

