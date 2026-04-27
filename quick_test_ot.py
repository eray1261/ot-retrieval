"""
Quick test to verify OT implementation works without downloading datasets.
Run this from your STRAP directory.
"""

import sys
import numpy as np

print("Testing OT implementation...")
print("=" * 60)

# Test 1: Import check
print("\n1. Checking imports...")
try:
    from strap.utils.retrieval_utils import compute_ot_match
    print("   ✓ Successfully imported compute_ot_match")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Basic OT matching
print("\n2. Testing basic OT matching...")
try:
    # Create fake trajectory embeddings (like DINOv2 would produce)
    query = np.random.randn(20, 128)  # 20 timesteps, 128-dim
    target = np.random.randn(50, 128)  # 50 timesteps
    
    start, end, cost = compute_ot_match(
        query_embedding=query,
        target_embedding=target,
        min_length=15,
        alpha=0.5,
        method='adaptive'
    )
    
    print(f"   Query length: {len(query)}")
    print(f"   Target length: {len(target)}")
    print(f"   Match found: [{start}, {end})")
    print(f"   Match length: {end - start}")
    print(f"   Cost: {cost:.4f}")
    
    if start >= 0 and end > start and cost < float('inf'):
        print("   ✓ OT matching works!")
    else:
        print("   ✗ OT returned invalid match")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ OT matching failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Different alpha values
print("\n3. Testing different alpha values...")
try:
    query = np.random.randn(15, 128)
    target = np.random.randn(40, 128)
    
    for alpha in [0.0, 0.5, 1.0]:
        _, _, cost = compute_ot_match(query, target, alpha=alpha)
        print(f"   alpha={alpha:.1f}: cost={cost:.4f}")
    
    print("   ✓ Alpha parameter works!")
    
except Exception as e:
    print(f"   ✗ Alpha test failed: {e}")
    sys.exit(1)

# Test 4: Edge case - query longer than target
print("\n4. Testing edge case (query > target)...")
try:
    query = np.random.randn(60, 128)  # Longer than target
    target = np.random.randn(40, 128)
    
    start, end, cost = compute_ot_match(query, target)
    
    if start == -1 and end == -1 and cost == float('inf'):
        print("   ✓ Correctly handled invalid case!")
    else:
        print("   ✗ Should have returned invalid match")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Edge case test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! OT implementation is working correctly.")
print("=" * 60)
print("\nYou can now:")
print("1. Download datasets: python data/download_libero.py")
print("2. Encode datasets: python strap/embedding/encode_datasets.py")
print("3. Run OT retrieval: python strap/retrieval/retrieval.py")
