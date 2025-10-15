#!/usr/bin/env python
"""
Minimal test to see if ProcgenGym3Env can be created at all
"""
import sys
import os

# Set environment variables BEFORE imports
print("=" * 60)
print("MINIMAL PROCGEN TEST")
print("=" * 60)

print("\n1. Setting threading environment variables...")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
print("   ✓ Threading variables set")

print("\n2. Removing MPI environment variables...")
removed_vars = []
for key in list(os.environ.keys()):
    if any(key.startswith(prefix) for prefix in ['OMPI_', 'PMI_', 'MPI_', 'SLURM_MPI', 'I_MPI_']):
        removed_vars.append(key)
        del os.environ[key]
if removed_vars:
    print(f"   ✓ Removed: {', '.join(removed_vars)}")
else:
    print("   ✓ No MPI variables found")

print("\n3. Blocking mpi4py module...")
class BlockMPI:
    def __getattr__(self, name):
        raise ImportError("mpi4py is intentionally blocked")
sys.modules['mpi4py'] = BlockMPI()
sys.modules['mpi4py.MPI'] = BlockMPI()
print("   ✓ mpi4py blocked")

print("\n4. Importing procgen...")
try:
    from procgen import ProcgenGym3Env
    print("   ✓ procgen imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import procgen: {e}")
    sys.exit(1)

print("\n5. Creating ProcgenGym3Env (this is where it usually hangs)...")
print("   Parameters:")
print("     - num=1")
print("     - env_name='coinrun'")
print("     - num_levels=1")
print("     - start_level=0")
print("     - distribution_mode='easy'")
print("     - num_threads=0")
print("     - render_mode=None")
print("     - rand_seed=12345")
print("\n   Attempting to create environment...")

import time
start_time = time.time()

try:
    env = ProcgenGym3Env(
        num=1,
        env_name='coinrun',
        num_levels=1,
        start_level=0,
        distribution_mode='easy',
        num_threads=0,
        render_mode=None,
        rand_seed=12345,
    )
    elapsed = time.time() - start_time
    print(f"   ✓ SUCCESS! Environment created in {elapsed:.2f} seconds")
    
    print("\n6. Testing environment reset...")
    obs = env.observe()
    print(f"   ✓ Environment reset successful")
    print(f"   Observation shape: {obs['rgb'].shape}")
    
    print("\n7. Testing environment step...")
    import numpy as np
    action = np.array([0])  # action 0
    env.act(action)
    reward, obs, first = env.observe()
    print(f"   ✓ Environment step successful")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nProcgen is working correctly on this system.")
    print("The issue must be elsewhere in the render.py script.")
    
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n   ✗ TIMEOUT/INTERRUPTED after {elapsed:.2f} seconds")
    print("\n" + "=" * 60)
    print("HUNG DURING ENVIRONMENT CREATION")
    print("=" * 60)
    print("\nThis confirms procgen hangs on this SLURM configuration.")
    print("Possible solutions:")
    print("  1. Try running on a different node")
    print("  2. Check if procgen C library is compatible with this system")
    print("  3. Try rebuilding procgen from source")
    print("  4. Use salloc instead of srun for interactive debugging")
    sys.exit(1)
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n   ✗ FAILED after {elapsed:.2f} seconds")
    print(f"   Error: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    print("ERROR DURING ENVIRONMENT CREATION")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)

