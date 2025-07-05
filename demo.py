#!/usr/bin/env python3
"""
Simple demonstration script for the GPT-2 model.
This script shows how to use the model for text generation.
"""

import torch
from gpt2_model import GPT2Model

def demo_model():
    """Demonstrate the GPT-2 model with a simple example"""
    print("🤖 Simple GPT-2 Model Demo")
    print("=" * 50)
    
    # Create a small model for demonstration
    print("Creating GPT-2 model...")
    model = GPT2Model(
        vocab_size=256,  # ASCII characters
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_seq_length=128,
        d_ff=1024
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {num_params:,} parameters")
    print()
    
    # Test with random input
    print("Testing forward pass...")
    batch_size = 2
    seq_length = 32
    
    # Generate random input tokens
    input_ids = torch.randint(32, 127, (batch_size, seq_length))  # Printable ASCII
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"✅ Input shape: {input_ids.shape}")
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Forward pass successful!")
    print()
    
    # Test text generation
    print("Testing text generation...")
    
    # Convert a simple text to tokens
    prompt = "Hello world"
    prompt_tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long)
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"📝 Prompt tokens: {prompt_tokens.tolist()[0]}")
    
    # Generate text
    with torch.no_grad():
        generated_tokens = model.generate(
            prompt_tokens, 
            max_new_tokens=20, 
            temperature=0.8,
            top_k=30
        )
    
    # Convert back to text
    generated_text = ''.join([chr(max(0, min(255, t))) for t in generated_tokens[0].tolist()])
    
    print(f"✅ Generated: '{generated_text}'")
    print(f"✅ Generated tokens: {generated_tokens[0].tolist()}")
    print()
    
    # Model summary
    print("📊 Model Summary:")
    print(f"   • Vocabulary size: {model.token_embedding.num_embeddings}")
    print(f"   • Hidden size: {model.d_model}")
    print(f"   • Number of layers: {len(model.transformer_blocks)}")
    print(f"   • Number of heads: {model.transformer_blocks[0].attention.num_heads}")
    print(f"   • Max sequence length: {model.max_seq_length}")
    print(f"   • Total parameters: {num_params:,}")
    print()
    
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    try:
        demo_model()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure you have PyTorch installed: pip install torch")