#!/usr/bin/env python3
"""
Quick test script to verify basic goin.py functionality without full training.
Tests critical components that could break from our code changes.
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

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 1)

    u, s, l = model.run(x)

    # Check output shapes
    assert u.shape == (batch_size, seq_len, 1), f"Expected u shape {(batch_size, seq_len, 1)}, got {u.shape}"
    assert s.shape == (batch_size, seq_len, 1), f"Expected s shape {(batch_size, seq_len, 1)}, got {s.shape}"
    assert l.shape[0] == batch_size and l.shape[1] == seq_len, f"Expected l shape starts with {(batch_size, seq_len)}, got {l.shape}"

    # Check no NaNs (except first prediction)
    assert not torch.isnan(u[:, 1:, :]).any(), "Predictions u[1:] should not contain NaNs"
    assert not torch.isnan(s[:, 1:, :]).any(), "Predictions s[1:] should not contain NaNs"
    assert not torch.isnan(l[:, 1:, :]).any(), "Predictions l[1:] should not contain NaNs"

    # First prediction should be NaN
    assert torch.isnan(u[:, 0, :]).all(), "First prediction u[0] should be NaN"

    print("✓ Model forward pass successful")
    return model

def test_generate_batch_integration():
    """Test that generate_batch integration works correctly."""
    print("Testing generate_batch integration...")

    gm = coin.GenerativeModel('validation')

    n_trials, batch_size = 50, 4
    y, _, c = gm.generate_batch(n_trials, batch_size)

    # Convert as done in goin.py
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

def test_compute_loss_method():
    """Test compute_loss method integration."""
    print("Testing compute_loss method...")

    model = goin.Model(n_hidden=4, n_layers=1, dev='cpu')
    gm = coin.GenerativeModel('validation')

    # Test compute_loss method
    loss_obs, loss_ctx, loss_dd = model.compute_loss(gm, n_trials=20, batch_size=2, ctx=True)

    # Check outputs
    assert isinstance(loss_obs, torch.Tensor), "Observation loss should be a tensor"
    assert isinstance(loss_ctx, torch.Tensor), "Context loss should be a tensor"
    assert isinstance(loss_dd, torch.Tensor), "Data loss should be a tensor"

    assert not torch.isnan(loss_obs), "Observation loss should not be NaN"
    assert not torch.isnan(loss_ctx), "Context loss should not be NaN"

    print("✓ compute_loss method successful")
    return loss_obs, loss_ctx, loss_dd

def main():
    """Run quick tests."""
    print("Starting quick goin.py functionality tests...\n")

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

        # Test 5: compute_loss integration
        losses = test_compute_loss_method()
        print()

        print("🎉 All quick tests passed! Basic goin.py functionality is working correctly.")
        print("\nChanges appear safe for basic operations.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Changes should NOT be committed until issues are resolved.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)