#!/usr/bin/env python3
"""Simple test script to verify device detection works correctly."""

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    # Test tensor creation and basic operations
    import torch
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(f"Tensor operation successful on {device}")
    print(f"Result shape: {z.shape}")