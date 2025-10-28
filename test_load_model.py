"""
Quick test to see if we can load the GitHub models
"""

import sys
from pathlib import Path
import d3rlpy

# Try different approaches to loading

print("=" * 80)
print("TEST 1: Try loading without any custom encoder definitions")
print("=" * 80)

model_path = Path("github_models/ddqn_online_att_model_final.d3")

try:
    model = d3rlpy.load_learnable(str(model_path))
    print(f"SUCCESS! Model type: {type(model).__name__}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "=" * 80)
print("TEST 2: Check what's actually stored in the model file")
print("=" * 80)

import torch

try:
    checkpoint = torch.load(str(model_path))
    print(f"Checkpoint keys: {checkpoint.keys()}")

    if 'params' in checkpoint:
        print(f"\nParams keys: {checkpoint['params'].keys()}")

        # Look for encoder info
        for key, value in checkpoint['params'].items():
            if 'encoder' in key.lower():
                print(f"  {key}: {value}")

except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "=" * 80)
print("TEST 3: Try loading with gym_sepsis available")
print("=" * 80)

# Make sure gym_sepsis can be imported
sys.path.insert(0, str(Path.cwd() / "gym-sepsis"))

try:
    import gym_sepsis
    print("gym_sepsis imported OK")

    model = d3rlpy.load_learnable(str(model_path))
    print(f"SUCCESS! Model type: {type(model).__name__}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
