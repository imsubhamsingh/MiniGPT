"""
Simple tokenizer for the language model.
"""

import json
import os
from typing import List, Dict, Optional


class SimpleTokenizer:
    """A simple character/word-level tokenizer."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Optional vocabulary dictionary {token: id}
        """
        if vocab is None:
            # Default vocabulary with special tokens
            self.vocab = {
                "<pad>": 0,
                "<unk>": 1,
                "<bos>": 2,
                "<eos>": 3,
            }
            self.vocab_size = 4
        else:
            self.vocab = vocab.copy()
            self.vocab_size = len(vocab)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.unk_token_id = self.vocab.get("<unk>", 1)
        self.bos_token_id = self.vocab.get("<bos>", 2)
        self.eos_token_id = self.vocab.get("<eos>", 3)
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a token to be included
        """
        # Count token frequencies (character-level)
        token_counts = {}
        for text in texts:
            for char in text:
                token_counts[char] = token_counts.get(char, 0) + 1
        
        # Build vocabulary
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Add tokens that meet minimum frequency
        for token, count in sorted(token_counts.items()):
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for char in text:
            token_ids.append(self.vocab.get(char, self.unk_token_id))
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                    continue
                tokens.append(token)
        
        return "".join(tokens)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer vocabulary."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer config
        config = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, tokenizer_directory: str):
        """Load tokenizer from saved directory."""
        vocab_path = os.path.join(tokenizer_directory, "vocab.json")
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        
        return cls(vocab=vocab)

