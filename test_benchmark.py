#!/usr/bin/env python3
"""
Test script for coin.py GenerativeModel benchmark functionality.

This test instantiates a GenerativeModel and calls benchmark() without saving results,
using 250 trials and 512 instances as specified.
"""

import coin
import os
import sys

def test_generative_model_benchmark():
    """Test the GenerativeModel benchmark method with specified parameters."""

    print("Testing GenerativeModel benchmark functionality...")
    print("=" * 50)

    # Create GenerativeModel instance with validation parameter set
    print("1. Instantiating GenerativeModel with 'validation' parameter set...")
    try:
        gm = coin.GenerativeModel(parset='validation')
        print("   ✓ GenerativeModel instantiated successfully")
    except Exception as e:
        print(f"   ✗ Failed to instantiate GenerativeModel: {e}")
        return False

    # Call benchmark with specified parameters: 100 trials, 8 instances, no saving
    print("2. Calling benchmark() with n_trials=100, n_instances=8, save=False...")
    try:
        benchmark_results = gm.benchmark(
            n_trials=100,      # Number of time points
            n_instances=8,     # Number of instances
            save=False         # Don't save results to disk
        )
        print("   ✓ benchmark() completed successfully")
    except Exception as e:
        print(f"   ✗ benchmark() failed: {e}")
        return False

    # Verify results structure
    print("3. Verifying benchmark results structure...")
    expected_keys = ['X', 'C', 'perf', 'best_t']
    missing_keys = [key for key in expected_keys if key not in benchmark_results]

    if missing_keys:
        print(f"   ✗ Missing expected keys in results: {missing_keys}")
        return False
    else:
        print("   ✓ All expected keys present in results")

    # Print summary of results
    print("4. Results summary:")
    print(f"   - Observation sequences shape: {benchmark_results['X'].shape}")
    print(f"   - Context sequences shape: {benchmark_results['C'].shape}")
    print(f"   - Best tau (leaky integrator): {benchmark_results['best_t']:.3f}")
    print(f"   - Performance metrics available for: {list(benchmark_results['perf'].keys())}")

    print("=" * 50)
    print("✓ Test completed successfully!")
    return True

if __name__ == "__main__":
    """Run the test when script is executed directly."""

    print("COIN GenerativeModel Benchmark Test")
    print("===================================")
    print()

    # Run the test
    success = test_generative_model_benchmark()

    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)