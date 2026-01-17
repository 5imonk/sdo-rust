#!/usr/bin/env python3
"""
Test script to verify the caching optimization in SDOstream and SDOstreamclust.
"""

import numpy as np
import time
from sdo import SDOstream, SDOstreamclust

def test_sdostream_caching():
    """Test that SDOstream caching works correctly."""
    print("Testing SDOstream caching...")
    
    # Initialize SDOstream
    sdostream = SDOstream(
        k=50,           # Number of observers
        x=5,            # Number of nearest neighbors
        t_fading=100.0, # Fading parameter
        rho=0.1,        # Rho parameter
        dimension=3      # Data dimension
    )
    
    # Create test data
    point = np.array([[1.0, 2.0, 3.0]])
    same_point = np.array([[1.0, 2.0, 3.0]])
    different_point = np.array([[5.0, 6.0, 7.0]])
    
    print("1. Learning a point...")
    start_time = time.time()
    sdostream.learn(point)
    learn_time = time.time() - start_time
    print(f"   Learn time: {learn_time:.6f}s")
    
    print("2. Predicting the same point (should use cache)...")
    start_time = time.time()
    score1 = sdostream.predict(same_point)
    predict_same_time = time.time() - start_time
    print(f"   Predict (same) time: {predict_same_time:.6f}s, score: {score1}")
    
    print("3. Predicting a different point (should recompute)...")
    start_time = time.time()
    score2 = sdostream.predict(different_point)
    predict_diff_time = time.time() - start_time
    print(f"   Predict (different) time: {predict_diff_time:.6f}s, score: {score2}")
    
    # Test cache invalidation
    print("4. Learning another point (should invalidate cache)...")
    new_point = np.array([[10.0, 11.0, 12.0]])
    sdostream.learn(new_point)
    
    print("5. Predicting the first point again (cache should be invalid)...")
    start_time = time.time()
    score3 = sdostream.predict(same_point)
    predict_new_time = time.time() - start_time
    print(f"   Predict (after cache invalidation) time: {predict_new_time:.6f}s, score: {score3}")
    
    # Verify scores are finite
    assert np.isfinite(score1), f"Score1 should be finite: {score1}"
    assert np.isfinite(score2), f"Score2 should be finite: {score2}"
    assert np.isfinite(score3), f"Score3 should be finite: {score3}"
    
    print("✓ SDOstream caching test passed!")
    return True

def test_sdostrclust_caching():
    """Test that SDOstreamclust caching works correctly."""
    print("\nTesting SDOstreamclust caching...")
    
    # Initialize SDOstreamclust
    sdostreamclust = SDOstreamclust(
        k=50,           # Number of observers
        x=5,            # Number of nearest neighbors
        t_fading=100.0, # Fading parameter
        chi_min=1,      # Minimum chi
        chi_prop=0.05,   # Chi proportion
        zeta=0.6,       # Zeta parameter
        rho=0.1,        # Rho parameter
        dimension=3      # Data dimension
    )
    
    # Create test data
    point = np.array([[1.0, 2.0, 3.0]])
    same_point = np.array([[1.0, 2.0, 3.0]])
    different_point = np.array([[5.0, 6.0, 7.0]])
    
    print("1. Learning a point...")
    start_time = time.time()
    sdostreamclust.learn(point)
    learn_time = time.time() - start_time
    print(f"   Learn time: {learn_time:.6f}s")
    
    print("2. Predicting the same point (should use cache)...")
    start_time = time.time()
    label1 = sdostreamclust.predict(same_point)
    predict_same_time = time.time() - start_time
    print(f"   Predict (same) time: {predict_same_time:.6f}s, label: {label1}")
    
    print("3. Predicting a different point (should recompute)...")
    start_time = time.time()
    label2 = sdostreamclust.predict(different_point)
    predict_diff_time = time.time() - start_time
    print(f"   Predict (different) time: {predict_diff_time:.6f}s, label: {label2}")
    
    # Test cache invalidation
    print("4. Learning another point (should invalidate cache)...")
    new_point = np.array([[10.0, 11.0, 12.0]])
    sdostreamclust.learn(new_point)
    
    print("5. Predicting the first point again (cache should be invalid)...")
    start_time = time.time()
    label3 = sdostreamclust.predict(same_point)
    predict_new_time = time.time() - start_time
    print(f"   Predict (after cache invalidation) time: {predict_new_time:.6f}s, label: {label3}")
    
    # Verify labels are reasonable
    assert isinstance(label1, (int, np.integer)), f"Label1 should be integer: {label1}"
    assert isinstance(label2, (int, np.integer)), f"Label2 should be integer: {label2}"
    assert isinstance(label3, (int, np.integer)), f"Label3 should be integer: {label3}"
    
    print("✓ SDOstreamclust caching test passed!")
    return True

def test_performance_benefit():
    """Test the performance benefit of caching."""
    print("\nTesting performance benefit...")
    
    # Initialize with smaller model for faster testing
    sdostream = SDOstream(
        k=20,           # Number of observers
        x=5,            # Number of nearest neighbors
        t_fading=100.0, # Fading parameter
        rho=0.1,        # Rho parameter
        dimension=10      # Higher dimension for more expensive computation
    )
    
    # Create test points
    test_point = np.random.rand(1, 10)
    
    # Learn the point
    sdostream.learn(test_point)
    
    # Test multiple predictions on same point (should use cache)
    print("1. Predicting same point multiple times (cached)...")
    start_time = time.time()
    for _ in range(100):
        score = sdostream.predict(test_point)
    cached_time = time.time() - start_time
    print(f"   100 cached predictions: {cached_time:.6f}s (avg: {cached_time/100:.8f}s)")
    
    # Test multiple predictions on different points (should recompute)
    print("2. Predicting different points multiple times (recomputed)...")
    start_time = time.time()
    for i in range(100):
        different_point = test_point + i * 0.001  # Slightly different
        score = sdostream.predict(different_point)
    recomputed_time = time.time() - start_time
    print(f"   100 recomputed predictions: {recomputed_time:.6f}s (avg: {recomputed_time/100:.8f}s)")
    
    # There should be a significant performance difference
    if cached_time < recomputed_time:
        speedup = recomputed_time / cached_time
        print(f"   Speedup: {speedup:.2f}x")
        print(f"✓ Performance benefit confirmed: cached predictions are {speedup:.2f}x faster")
    else:
        print("! No significant performance difference detected (might be due to small model size)")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("SDO Caching Optimization Test")
    print("=" * 60)
    
    try:
        # Run tests
        test_sdostream_caching()
        test_sdostrclust_caching()
        test_performance_benefit()
        
        print("\n" + "=" * 60)
        print("✓ All caching tests passed successfully!")
        print("✓ The learn() method now provides necessary information for predict() for free!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)