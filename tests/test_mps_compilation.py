#!/usr/bin/env python3
"""
Comprehensive MPS Compilation Test for HRM Models

This script tests torch.compile compatibility with different HRM model configurations
on Apple Silicon MPS devices. It helps identify which configurations work with
compilation and which ones fail.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    config: Dict[str, Any]
    compilation_success: bool
    forward_success: bool
    backward_success: bool
    error_message: str = ""
    compilation_time: float = 0.0
    inference_time: float = 0.0


def get_test_configurations() -> List[Tuple[str, Dict[str, Any]]]:
    """Get various model configurations to test."""
    
    # Base configuration
    base_config = {
        'batch_size': 2,
        'seq_len': 32,
        'vocab_size': 128,
        'num_puzzle_identifiers': 100,
        'puzzle_emb_ndim': 64,
        'hidden_size': 64,
        'expansion': 2.0,
        'num_heads': 4,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'float32'
    }
    
    configs = []
    
    # Test 1: Minimal configuration
    minimal = base_config.copy()
    minimal.update({
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 1,
        'L_layers': 1,
        'halt_max_steps': 2,
        'pos_encodings': 'rope'
    })
    configs.append(("Minimal (1 cycle, 1 layer)", minimal))
    
    # Test 2: Small configuration
    small = base_config.copy()
    small.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 4,
        'pos_encodings': 'rope'
    })
    configs.append(("Small (2 cycles, 2 layers)", small))
    
    # Test 3: Medium configuration
    medium = base_config.copy()
    medium.update({
        'H_cycles': 4,
        'L_cycles': 4,
        'H_layers': 4,
        'L_layers': 4,
        'halt_max_steps': 8,
        'pos_encodings': 'rope'
    })
    configs.append(("Medium (4 cycles, 4 layers)", medium))
    
    # Test 4: With learned positional encodings
    learned_pos = base_config.copy()
    learned_pos.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 4,
        'pos_encodings': 'learned'
    })
    configs.append(("Learned Positional Encodings", learned_pos))
    
    # Test 5: Large hidden size
    large_hidden = base_config.copy()
    large_hidden.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 4,
        'hidden_size': 256,
        'num_heads': 8,
        'pos_encodings': 'rope'
    })
    configs.append(("Large Hidden Size (256)", large_hidden))
    
    # Test 6: Many attention heads
    many_heads = base_config.copy()
    many_heads.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 4,
        'hidden_size': 128,
        'num_heads': 16,
        'pos_encodings': 'rope'
    })
    configs.append(("Many Attention Heads (16)", many_heads))
    
    # Test 7: Large sequence length
    long_seq = base_config.copy()
    long_seq.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 4,
        'seq_len': 128,
        'pos_encodings': 'rope'
    })
    configs.append(("Long Sequence (128)", long_seq))
    
    # Test 8: Complex configuration (similar to actual training)
    complex_config = base_config.copy()
    complex_config.update({
        'H_cycles': 8,
        'L_cycles': 8,
        'H_layers': 6,
        'L_layers': 6,
        'halt_max_steps': 16,
        'hidden_size': 128,
        'num_heads': 8,
        'seq_len': 64,
        'pos_encodings': 'rope'
    })
    configs.append(("Complex (8 cycles, 6 layers)", complex_config))
    
    # Test 9: No puzzle embeddings
    no_puzzle = base_config.copy()
    no_puzzle.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 4,
        'puzzle_emb_ndim': 0,  # Disable puzzle embeddings
        'pos_encodings': 'rope'
    })
    configs.append(("No Puzzle Embeddings", no_puzzle))
    
    # Test 10: Maximum halting steps
    max_halt = base_config.copy()
    max_halt.update({
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'halt_max_steps': 32,  # Very high
        'pos_encodings': 'rope'
    })
    configs.append(("Maximum Halting Steps (32)", max_halt))
    
    return configs


def test_model_configuration(name: str, config: Dict[str, Any], device: str = "mps") -> TestResult:
    """Test a single model configuration."""
    print(f"\nTesting: {name}")
    print("-" * 40)
    
    result = TestResult(name=name, config=config, 
                        compilation_success=False, 
                        forward_success=False, 
                        backward_success=False)
    
    try:
        # Import model components
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        from models.losses import ACTLossHead
        
        # Create model
        print("  Creating model...")
        with torch.device(device):
            model = HierarchicalReasoningModel_ACTV1(config)
            model = ACTLossHead(model, loss_type='softmax_cross_entropy')
            model = model.to(device)
        
        # Try compilation
        print("  Attempting compilation...")
        compilation_start = time.time()
        try:
            # Try different backends for MPS
            compiled_model = torch.compile(model, backend="aot_eager", dynamic=False)
            result.compilation_time = time.time() - compilation_start
            result.compilation_success = True
            print(f"  ✓ Compilation successful ({result.compilation_time:.2f}s)")
            model = compiled_model
        except Exception as e:
            result.error_message = str(e)[:200]
            print(f"  ✗ Compilation failed: {result.error_message}")
            print("    Continuing with uncompiled model...")
        
        # Test forward pass
        print("  Testing forward pass...")
        batch = {
            'inputs': torch.randint(0, config['vocab_size'], 
                                   (config['batch_size'], config['seq_len']), 
                                   device=device),
            'puzzle_identifiers': torch.randint(0, config['num_puzzle_identifiers'], 
                                               (config['batch_size'],), 
                                               device=device),
            'labels': torch.randint(0, config['vocab_size'], 
                                   (config['batch_size'], config['seq_len']), 
                                   device=device)
        }
        
        try:
            carry = model.initial_carry(batch)
            
            # Warm-up run
            _, _, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
            
            # Timed run
            inference_start = time.time()
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            result.inference_time = time.time() - inference_start
            
            result.forward_success = True
            print(f"  ✓ Forward pass successful (loss: {loss.item():.4f}, time: {result.inference_time:.4f}s)")
        except Exception as e:
            result.error_message = f"Forward failed: {str(e)[:200]}"
            print(f"  ✗ Forward pass failed: {result.error_message}")
            return result
        
        # Test backward pass
        print("  Testing backward pass...")
        try:
            loss.backward()
            result.backward_success = True
            print(f"  ✓ Backward pass successful")
        except Exception as e:
            result.error_message = f"Backward failed: {str(e)[:200]}"
            print(f"  ✗ Backward pass failed: {result.error_message}")
        
    except Exception as e:
        result.error_message = f"Model creation failed: {str(e)[:200]}"
        print(f"  ✗ Error: {result.error_message}")
    
    return result


def test_different_loss_types(device: str = "mps") -> List[TestResult]:
    """Test different loss configurations."""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT LOSS TYPES")
    print("=" * 60)
    
    base_config = {
        'batch_size': 2,
        'seq_len': 32,
        'vocab_size': 128,
        'num_puzzle_identifiers': 100,
        'puzzle_emb_ndim': 64,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 64,
        'expansion': 2.0,
        'num_heads': 4,
        'halt_max_steps': 4,
        'pos_encodings': 'rope',
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'float32'
    }
    
    loss_types = ['softmax_cross_entropy', 'stablemax_cross_entropy']
    results = []
    
    for loss_type in loss_types:
        print(f"\nTesting loss type: {loss_type}")
        print("-" * 40)
        
        result = TestResult(name=f"Loss: {loss_type}", config=base_config,
                           compilation_success=False, 
                           forward_success=False,
                           backward_success=False)
        
        try:
            from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
            from models.losses import ACTLossHead
            
            with torch.device(device):
                model = HierarchicalReasoningModel_ACTV1(base_config)
                model = ACTLossHead(model, loss_type=loss_type)
                model = model.to(device)
            
            # Try compilation
            try:
                compiled_model = torch.compile(model, dynamic=False)
                result.compilation_success = True
                print(f"  ✓ Compilation successful with {loss_type}")
                model = compiled_model
            except Exception as e:
                result.error_message = str(e)[:200]
                print(f"  ✗ Compilation failed with {loss_type}")
            
            # Test forward/backward
            batch = {
                'inputs': torch.randint(0, 128, (2, 32), device=device),
                'puzzle_identifiers': torch.randint(0, 100, (2,), device=device),
                'labels': torch.randint(0, 128, (2, 32), device=device)
            }
            
            carry = model.initial_carry(batch)
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            result.forward_success = True
            
            loss.backward()
            result.backward_success = True
            
            print(f"  ✓ Forward/backward successful with {loss_type}")
            
        except Exception as e:
            result.error_message = str(e)[:200]
            print(f"  ✗ Error with {loss_type}: {result.error_message}")
        
        results.append(result)
    
    return results


def main():
    """Run all MPS compilation tests."""
    print("=" * 60)
    print("MPS COMPILATION TEST SUITE FOR HRM MODELS")
    print("=" * 60)
    
    # Check device availability
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("ERROR: MPS is not available on this system.")
        print("This test requires an Apple Silicon Mac with PyTorch MPS support.")
        sys.exit(1)
    
    device = "mps"
    print(f"Running tests on: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run configuration tests
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT MODEL CONFIGURATIONS")
    print("=" * 60)
    
    configs = get_test_configurations()
    config_results = []
    
    for name, config in configs:
        result = test_model_configuration(name, config, device)
        config_results.append(result)
    
    # Run loss type tests
    loss_results = test_different_loss_types(device)
    
    # Combine all results
    all_results = config_results + loss_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPILATION TEST SUMMARY")
    print("=" * 60)
    
    compilation_success = sum(1 for r in all_results if r.compilation_success)
    forward_success = sum(1 for r in all_results if r.forward_success)
    backward_success = sum(1 for r in all_results if r.backward_success)
    total = len(all_results)
    
    print(f"\nOverall Results:")
    print(f"  Compilation succeeded: {compilation_success}/{total} ({100*compilation_success/total:.1f}%)")
    print(f"  Forward pass succeeded: {forward_success}/{total} ({100*forward_success/total:.1f}%)")
    print(f"  Backward pass succeeded: {backward_success}/{total} ({100*backward_success/total:.1f}%)")
    
    print("\nDetailed Results:")
    print("-" * 60)
    print(f"{'Configuration':<40} {'Compile':<10} {'Forward':<10} {'Backward':<10}")
    print("-" * 60)
    
    for result in all_results:
        compile_str = "✓" if result.compilation_success else "✗"
        forward_str = "✓" if result.forward_success else "✗"
        backward_str = "✓" if result.backward_success else "✗"
        
        # Add timing info if compilation succeeded
        if result.compilation_success and result.compilation_time > 0:
            compile_str += f" ({result.compilation_time:.1f}s)"
        
        print(f"{result.name:<40} {compile_str:<10} {forward_str:<10} {backward_str:<10}")
    
    # Identify patterns
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if compilation_success == total:
        print("✓ EXCELLENT: All model configurations compile successfully on MPS!")
        print("  torch.compile appears to be fully functional for HRM models.")
    elif compilation_success > 0:
        print(f"⚠ PARTIAL SUCCESS: {compilation_success}/{total} configurations compile on MPS")
        print("\nConfigurations that FAILED compilation:")
        for result in all_results:
            if not result.compilation_success:
                print(f"  • {result.name}")
                if result.error_message:
                    print(f"    Error: {result.error_message[:100]}...")
    else:
        print("✗ NO SUCCESS: torch.compile does not work with any tested configuration")
        print("  MPS compilation may not be supported in your PyTorch version")
    
    # Performance comparison if we have successful compilations
    if compilation_success > 0:
        print("\n" + "=" * 60)
        print("PERFORMANCE IMPACT")
        print("=" * 60)
        
        compiled_times = [r.inference_time for r in all_results 
                         if r.compilation_success and r.inference_time > 0]
        if compiled_times:
            avg_time = sum(compiled_times) / len(compiled_times)
            print(f"Average inference time for compiled models: {avg_time:.4f}s")
            print("Note: First run includes JIT compilation overhead")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if compilation_success == total:
        print("• MPS compilation is working well - it's enabled by default for training:")
        print("  python pretrain.py ...")
    elif compilation_success > total / 2:
        print("• MPS compilation works for most configs - it's enabled by default:")
        print("  python pretrain.py ...")
        print("• If compilation fails, training will continue uncompiled")
    else:
        print("• MPS compilation has limited support - use with caution")
        print("• Consider upgrading PyTorch for better MPS support")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return compilation_success == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)