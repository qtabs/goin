#!/usr/bin/env python3
"""
Test script to verify goin.py functionality after code quality improvements.
Tests model instantiation, training, and prediction capabilities.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path to import local modules
sys.path.insert(0, os.getcwd())

import goin
import coin

def test_model_instantiation():
    """Test that Model can be instantiated with small parameters."""
    print("Testing model instantiation...")

    # Use small parameters for fast processing
    model = goin.Model(n_hidden=8, n_layers=1, dev='cpu')

    assert model.model is not None, "Model should be instantiated"
    assert model.dev == 'cpu', "Device should be set correctly"
    assert model.model.n_hidden == 8, "Hidden size should be set correctly"

    print("✓ Model instantiation successful")
    return model

def test_model_forward_pass():
    """Test that model can perform forward pass."""
    print("Testing model forward pass...")

    model = goin.Model(n_hidden=8, n_layers=1, dev='cpu')

    # Create small test input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 1)

    # Test forward pass
    u, s, l = model.run(x)

    # Check output shapes
    assert u.shape == (batch_size, seq_len, 1), f"Expected u shape {(batch_size, seq_len, 1)}, got {u.shape}"
    assert s.shape == (batch_size, seq_len, 1), f"Expected s shape {(batch_size, seq_len, 1)}, got {s.shape}"
    assert l.shape[0] == batch_size and l.shape[1] == seq_len, f"Expected l shape starts with {(batch_size, seq_len)}, got {l.shape}"

    # Check no NaNs (except first prediction which is expected to be NaN)
    assert not torch.isnan(u[:, 1:, :]).any(), "Predictions u[1:] should not contain NaNs"
    assert not torch.isnan(s[:, 1:, :]).any(), "Predictions s[1:] should not contain NaNs"
    assert not torch.isnan(l[:, 1:, :]).any(), "Predictions l[1:] should not contain NaNs"

    # First prediction should be NaN (cannot predict without prior observation)
    assert torch.isnan(u[:, 0, :]).all(), "First prediction u[0] should be NaN"

    print("✓ Model forward pass successful")
    return model

def test_generate_batch_integration():
    """Test that generate_batch integration works correctly."""
    print("Testing generate_batch integration...")

    # Create GenerativeModel with valid parset
    gm = coin.GenerativeModel('validation')

    # Test generate_batch call and tensor conversion
    n_trials, batch_size = 50, 4
    y, _, c = gm.generate_batch(n_trials, batch_size)

    # Convert as done in goin.py (y and c come as 2D, need to add 3rd dimension)
    x = torch.tensor(y, dtype=torch.float, requires_grad=False).to('cpu').unsqueeze(-1)
    c_tensor = torch.tensor(c, dtype=torch.long, requires_grad=False).to('cpu').unsqueeze(-1)

    # Check shapes
    assert x.shape == (batch_size, n_trials, 1), f"Expected x shape {(batch_size, n_trials, 1)}, got {x.shape}"
    assert c_tensor.shape == (batch_size, n_trials, 1), f"Expected c shape {(batch_size, n_trials, 1)}, got {c_tensor.shape}"

    # Check no NaNs
    assert not torch.isnan(x).any(), "Generated observations should not contain NaNs"
    assert not torch.isnan(c_tensor.float()).any(), "Generated contexts should not contain NaNs"

    print("✓ generate_batch integration successful")
    return gm, x, c_tensor

def test_training_basic():
    """Test basic training functionality with minimal parameters."""
    print("Testing basic training...")

    # Create model with minimal parameters
    model = goin.Model(n_hidden=4, n_layers=1, dev='cpu')

    # Test training with very small parameters
    try:
        # Minimal training: 1 epoch, 2 batches
        model.train(
            parset='validation',
            savename='test_model',
            oracle=0.5,
            lr=0.01,
            train_sched=(1, 2),  # 1 epoch, 2 batches
            resume=False
        )
        print("✓ Training completed without errors")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise

    return model

def test_benchmark():
    """Test benchmark functionality."""
    print("Testing benchmark functionality...")

    model = goin.Model(n_hidden=4, n_layers=1, dev='cpu')

    # Create small test data
    batch_size, n_trials = 3, 20
    x = torch.randn(batch_size, n_trials, 1).numpy()
    c = torch.randint(0, 3, (batch_size, n_trials, 1)).numpy()

    # Test benchmark
    perf, u, s, c_hat, c_pr = model.benchmark(x, c)

    # Check outputs
    assert isinstance(perf, dict), "Performance should be a dictionary"
    assert u.shape == x.shape, f"Predictions u should match input shape {x.shape}, got {u.shape}"
    assert s.shape == x.shape, f"Predictions s should match input shape {x.shape}, got {s.shape}"
    assert c_hat.shape == (batch_size, n_trials), f"Context predictions should have shape {(batch_size, n_trials)}, got {c_hat.shape}"

    # Check no NaNs in predictions (except first timestep)
    assert not np.isnan(u[:, 1:, :]).any(), "Benchmark predictions u[1:] should not contain NaNs"
    assert not np.isnan(s[:, 1:, :]).any(), "Benchmark predictions s[1:] should not contain NaNs"

    print("✓ Benchmark functionality successful")
    return perf, u, s, c_hat, c_pr

def test_ctxlossfunc():
    """Test context loss function with proper tensor shapes."""
    print("Testing context loss function...")

    model = goin.Model(n_hidden=4, n_layers=1, dev='cpu')

    # Create test data with correct shapes
    batch_size, n_trials, n_contexts = 2, 10, 3
    c = torch.randint(0, n_contexts, (batch_size, n_trials, 1), dtype=torch.long)
    l = torch.randn(batch_size, n_trials, n_contexts)

    # Test context loss function
    loss = model.ctxlossfunc(c, l)

    # Check output
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.numel() == 1, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("✓ Context loss function successful")
    return loss

def main():
    """Run all tests."""
    print("Starting goin.py functionality tests...\n")

    try:
        # Test 1: Model instantiation
        model = test_model_instantiation()
        print()

        # Test 2: Forward pass
        model = test_model_forward_pass()
        print()

        # Test 3: generate_batch integration
        gm, x, c = test_generate_batch_integration()
        print()

        # Test 4: Context loss function
        loss = test_ctxlossfunc()
        print()

        # Test 5: Benchmark functionality
        perf, u, s, c_hat, c_pr = test_benchmark()
        print()

        # Test 6: Basic training (this takes longer)
        print("Warning: Training test may take 30-60 seconds...")
        trained_model = test_training_basic()
        print()

        print("🎉 All tests passed! goin.py functionality is working correctly.")
        print("\nChanges are safe to commit.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Changes should NOT be committed until issues are resolved.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)