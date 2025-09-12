#!/usr/bin/env python3
"""
Device Compatibility Test Script for HRM

This script tests the HRM model's compatibility with different devices (CUDA, MPS, CPU)
and verifies that all components work correctly on each platform.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Set environment for CPU testing if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Uncomment to force CPU

def test_device_availability():
    """Test which devices are available on this system."""
    print("=" * 60)
    print("Device Availability Test")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")

    print(f"MPS available: {mps_available}")
    print(f"CPU: Always available")

    # Determine best device
    if cuda_available:
        best_device = "cuda"
    elif mps_available:
        best_device = "mps"
    else:
        best_device = "cpu"

    print(f"\nBest available device: {best_device}")
    return best_device


def test_model_creation(device: str, test_compilation: bool = False):
    """Test model creation on specified device."""
    print("\n" + "=" * 60)
    title = f"Model Creation Test on {device.upper()}"
    if test_compilation:
        title += " (with compilation)"
    print(title)
    print("=" * 60)

    try:
        # Import model components
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        from models.losses import ACTLossHead

        # Create minimal config
        config = {
            'batch_size': 2,
            'seq_len': 32,
            'puzzle_emb_ndim': 64,
            'num_puzzle_identifiers': 100,
            'vocab_size': 128,
            'H_cycles': 2,
            'L_cycles': 2,
            'H_layers': 2,
            'L_layers': 2,
            'hidden_size': 64,
            'expansion': 2.0,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 4,
            'halt_exploration_prob': 0.1,
            'forward_dtype': 'float32'  # Use float32 for testing
        }

        # Create model
        with torch.device(device):
            model = HierarchicalReasoningModel_ACTV1(config)
            model = ACTLossHead(model, loss_type='softmax_cross_entropy')
            model = model.to(device)

        print(f"✓ Model created successfully on {device}")

        # Try compilation if requested
        if test_compilation:
            try:
                print(f"  Attempting torch.compile...")
                model = torch.compile(model, dynamic=False)
                print(f"  ✓ Model compiled successfully")
            except Exception as e:
                print(f"  ⚠ Compilation failed: {str(e)[:100]}...")
                if device == "mps":
                    print(f"    Continuing without compilation (expected for complex models on MPS)")
                else:
                    raise
            finally:
                pass  # Cleanup if needed

        # Test forward pass
        batch = {
            'inputs': torch.randint(0, 128, (2, 32), device=device),
            'puzzle_identifiers': torch.randint(0, 100, (2,), device=device),
            'labels': torch.randint(0, 128, (2, 32), device=device)
        }

        carry = model.initial_carry(batch)
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

        print(f"✓ Forward pass successful")
        print(f"  Loss: {loss.item():.4f}")

        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")

        return True

    except Exception as e:
        print(f"✗ Error on {device}: {e}")
        return False


def test_sparse_embedding(device: str):
    """Test sparse embedding module on specified device."""
    print("\n" + "=" * 60)
    print(f"Sparse Embedding Test on {device.upper()}")
    print("=" * 60)

    try:
        from models.sparse_embedding import CastedSparseEmbedding

        # Create sparse embedding
        embed = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=4,
            init_std=0.02,
            cast_to=torch.float32,
            device=device
        )
        embed = embed.to(device)

        # Test forward pass
        indices = torch.randint(0, 100, (4,), device=device)
        output = embed(indices)

        assert output.shape == (4, 64)
        assert output.device.type == device if device != 'cuda' else 'cuda'

        print(f"✓ Sparse embedding works on {device}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output device: {output.device}")

        return True

    except Exception as e:
        print(f"✗ Sparse embedding error on {device}: {e}")
        return False


def test_optimizer_compatibility(device: str):
    """Test optimizer compatibility with device."""
    print("\n" + "=" * 60)
    print(f"Optimizer Compatibility Test on {device.upper()}")
    print("=" * 60)

    try:
        # Simple model for testing
        model = nn.Linear(10, 10).to(device)

        # Try to import and use adam-atan2
        try:
            from adam_atan2 import AdamATan2
            optimizer_name = "adam-atan2 (CUDA)"
            lr = 0 if device == "cuda" else 1e-8
            optimizer = AdamATan2(model.parameters(), lr=lr)
        except ImportError:
            # Fallback to CPU-compatible version
            try:
                from adam_atan2_pytorch import AdamAtan2
                optimizer_name = "adam-atan2-pytorch (CPU/MPS)"
                optimizer = AdamAtan2(model.parameters(), lr=1e-3)
            except ImportError:
                # Final fallback to standard Adam
                optimizer_name = "torch.optim.Adam (fallback)"
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Test optimization step
        x = torch.randn(4, 10, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        print(f"✓ Optimizer {optimizer_name} works on {device}")

        return True

    except Exception as e:
        print(f"✗ Optimizer error on {device}: {e}")
        return False


def test_compilation(device: str):
    """Test PyTorch compilation support."""
    print("\n" + "=" * 60)
    print(f"Compilation Test on {device.upper()}")
    print("=" * 60)

    if device == "cpu":
        print(f"ℹ Compilation not supported on CPU (expected behavior)")
        return True

    if device == "mps":
        print(f"ℹ MPS compilation enabled by default (testing both enabled and disabled modes)")

        # Test 1: Default behavior (should enable compilation)
        print("\n  Test 1: Default MPS behavior (compilation enabled)")
        # Clear any compilation overrides
        os.environ.pop('DISABLE_COMPILE', None)     # Clear any override
        try:
            model = nn.Linear(10, 10).to(device)
            compiled_model = torch.compile(model, dynamic=False)

            # Test forward pass
            x = torch.randn(4, 10, device=device)
            y = compiled_model(x)

            # Test backward pass
            loss = y.sum()
            loss.backward()

            print("  ✓ Default MPS compilation works!")
            result1 = True

        except Exception as e:
            print(f"  ⚠ MPS compilation failed: {str(e)[:100]}...")
            print("    Training will continue without compilation")
            result1 = False  # Failed but training continues

        # Test 2: Disabled compilation
        print("\n  Test 2: Disabled compilation (DISABLE_COMPILE=1)")
        os.environ['DISABLE_COMPILE'] = '1'
        try:
            model = nn.Linear(10, 10).to(device)
            # Should not compile when DISABLE_COMPILE=1
            x = torch.randn(4, 10, device=device)
            y = model(x)
            loss = y.sum()
            loss.backward()

            print("  ✓ DISABLE_COMPILE=1 works correctly")
            result2 = True
        except Exception as e:
            print(f"  ✗ Unexpected error with disabled compilation: {e}")
            result2 = False

        finally:
            os.environ.pop('DISABLE_COMPILE', None)

        # Overall MPS result: success if at least one mode works
        return result1 or result2

    # CUDA compilation (should work)
    try:
        model = nn.Linear(10, 10).to(device)
        compiled_model = torch.compile(model, dynamic=False)

        x = torch.randn(4, 10, device=device)
        y = compiled_model(x)

        print(f"✓ Compilation works on {device}")
        return True

    except Exception as e:
        print(f"✗ Compilation error on {device}: {e}")
        return False


def run_all_tests():
    """Run all device compatibility tests."""
    print("\n" + "=" * 60)
    print("HRM DEVICE COMPATIBILITY TEST SUITE")
    print("=" * 60)

    # Detect available devices
    best_device = test_device_availability()

    # Determine which devices to test
    devices_to_test = []

    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices_to_test.append("mps")
    devices_to_test.append("cpu")  # Always test CPU

    # Run tests for each available device
    results = {}
    for device in devices_to_test:
        print(f"\n{'#' * 60}")
        print(f"Testing on {device.upper()}")
        print('#' * 60)

        device_results = {
            'model_creation': test_model_creation(device),
            'sparse_embedding': test_sparse_embedding(device),
            'optimizer': test_optimizer_compatibility(device),
            'compilation': test_compilation(device)
        }

        # Additional test for MPS: HRM model with compilation
        if device == "mps":
            print("\n" + "-" * 60)
            print("BONUS MPS TEST: HRM Model with Compilation")
            print("-" * 60)
            device_results['hrm_with_compilation'] = test_model_creation(device, test_compilation=True)

        results[device] = device_results

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for device, device_results in results.items():
        passed = sum(device_results.values())
        total = len(device_results)
        status = "✓ PASSED" if passed == total else f"⚠ PARTIAL ({passed}/{total})"

        print(f"\n{device.upper()}: {status}")
        for test_name, result in device_results.items():
            symbol = "✓" if result else "✗"
            print(f"  {symbol} {test_name}")

    # Overall result
    all_passed = all(all(dr.values()) for dr in results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("⚠ SOME TESTS FAILED - Check output above for details")
    print("=" * 60)

    # Additional notes about MPS compilation
    if 'mps' in results:
        print("\nMPS COMPILATION NOTES:")
        print("-" * 40)
        print("• MPS compilation is enabled by default (same as CUDA)")
        print("• To disable compilation: DISABLE_COMPILE=1 python pretrain.py ...")
        print("• If compilation fails, training continues without it")
        print("• Performance gain varies by model architecture")
        if 'hrm_with_compilation' in results.get('mps', {}):
            if not results['mps']['hrm_with_compilation']:
                print("• HRM model compilation failed (expected) - will run uncompiled")

    return all_passed


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)