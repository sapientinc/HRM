# HRM Test Suite

This directory contains diagnostic and compatibility tests for the Hierarchical Reasoning Model (HRM) implementation.

## Test Files

### Device Compatibility Tests

#### `test_device_compatibility.py`
General device compatibility testing across CUDA, MPS, and CPU devices.

**Purpose:** Verify that HRM models work correctly on different hardware accelerators.

**What it tests:**
- Device detection (CUDA/MPS/CPU)
- Model creation and initialization
- Forward and backward passes
- Sparse embedding functionality
- Optimizer compatibility
- PyTorch compilation support

**Usage:**
```bash
python tests/test_device_compatibility.py
```

#### `test_cuda_compatibility.py`
CUDA-specific compatibility testing.

**Purpose:** Ensure CUDA-specific optimizations and features work correctly.

**Usage:**
```bash
python tests/test_cuda_compatibility.py
```

### MPS Compilation Testing

#### `test_mps_compilation.py`
Comprehensive testing of PyTorch compilation support on Apple Silicon (MPS).

**Purpose:** Test which HRM model configurations successfully compile with `torch.compile` on MPS devices.

**What it tests:**
- 10+ different model configurations
- Various model sizes (`hidden_size`, layers, cycles)
- Different loss types (`softmax_cross_entropy`, `stablemax_cross_entropy`)
- Different positional encodings (RoPE vs learned)
- Performance impact of compilation

**Usage:**
```bash
python tests/test_mps_compilation.py
```

**Output:**
- Success rate for different configurations
- Specific errors for failed compilations
- Recommendations based on test results

## Running All Tests

To run all compatibility tests:
```bash
# Run all tests
for test in tests/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

## When to Run These Tests

Run these tests when:
- Setting up HRM on a new system
- After updating PyTorch or CUDA versions
- Debugging device-specific issues
- Verifying MPS compilation compatibility
- Before deploying to different hardware

## Notes

- These tests are diagnostic tools, not unit tests
- They help identify hardware/software compatibility issues
- Results may vary based on PyTorch version and hardware
- MPS compilation support requires PyTorch 2.8.0+
- CUDA tests require NVIDIA GPU with appropriate drivers
