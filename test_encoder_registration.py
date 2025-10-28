"""
Test different methods of encoder registration in d3rlpy
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Sequence

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import d3rlpy
from d3rlpy.models.encoders import Encoder, EncoderFactory
from d3rlpy.types import Shape

print("=" * 80)
print("D3RLPY ENCODER REGISTRATION DIAGNOSIS")
print("=" * 80)
print(f"\nd3rlpy version: {d3rlpy.__version__}")
print()

# ============================================================================
# Define minimal encoder for testing
# ============================================================================

class TestEncoder(Encoder):
    """Minimal test encoder"""

    def __init__(self, observation_shape: Shape):
        super().__init__()
        self.observation_shape = observation_shape
        input_size = observation_shape[0] if isinstance(observation_shape[0], int) else observation_shape[0][0]
        self._feature_size = 128
        self.fc = nn.Linear(input_size, self._feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_feature_size(self) -> int:
        return self._feature_size


class TestEncoderFactory(EncoderFactory):
    """Factory for TestEncoder"""

    def create(self, observation_shape: Shape) -> TestEncoder:
        return TestEncoder(observation_shape)


# ============================================================================
# Test 1: Check what's available in d3rlpy.models.encoders
# ============================================================================

print("TEST 1: Check d3rlpy.models.encoders attributes")
print("-" * 80)

import d3rlpy.models.encoders as enc_module

# List all public attributes
public_attrs = [attr for attr in dir(enc_module) if not attr.startswith('_')]
print(f"Public attributes: {public_attrs}")
print()

# Check for specific attributes we need
checks = {
    'register_encoder_factory': hasattr(enc_module, 'register_encoder_factory'),
    'CONFIG_LIST': hasattr(enc_module, 'CONFIG_LIST'),
    'Encoder': hasattr(enc_module, 'Encoder'),
    'EncoderFactory': hasattr(enc_module, 'EncoderFactory'),
}

for attr, exists in checks.items():
    status = 'EXISTS' if exists else 'NOT FOUND'
    print(f"  {attr}: {status}")
print()


# ============================================================================
# Test 2: Try register_encoder_factory if available
# ============================================================================

if hasattr(enc_module, 'register_encoder_factory'):
    print("TEST 2: Try using register_encoder_factory()")
    print("-" * 80)

    try:
        enc_module.register_encoder_factory(TestEncoderFactory, 'test_encoder')
        print("[OK] register_encoder_factory() succeeded!")
        print()

        # Now try to verify it was registered
        print("Verifying registration...")

        # Check CONFIG_LIST if available
        if hasattr(enc_module, 'CONFIG_LIST'):
            if 'test_encoder' in enc_module.CONFIG_LIST:
                print("[OK] 'test_encoder' found in CONFIG_LIST")
            else:
                print("[ERROR] 'test_encoder' NOT in CONFIG_LIST")
                print(f"  Available keys: {list(enc_module.CONFIG_LIST.keys())}")
        else:
            print("  (CONFIG_LIST not available for verification)")

        print()

    except Exception as e:
        print(f"[ERROR] register_encoder_factory() failed: {e}")
        import traceback
        traceback.print_exc()
        print()
else:
    print("TEST 2: SKIPPED - register_encoder_factory not available")
    print()


# ============================================================================
# Test 3: Try direct CONFIG_LIST access
# ============================================================================

if hasattr(enc_module, 'CONFIG_LIST'):
    print("TEST 3: Try direct CONFIG_LIST access")
    print("-" * 80)

    try:
        print(f"CONFIG_LIST type: {type(enc_module.CONFIG_LIST)}")
        print(f"Current keys: {list(enc_module.CONFIG_LIST.keys())}")
        print()

        # Try to add our encoder
        enc_module.CONFIG_LIST['test_encoder_direct'] = TestEncoderFactory
        print("[OK] Direct assignment succeeded!")
        print(f"Updated keys: {list(enc_module.CONFIG_LIST.keys())}")
        print()

    except Exception as e:
        print(f"[ERROR] Direct CONFIG_LIST access failed: {e}")
        import traceback
        traceback.print_exc()
        print()
else:
    print("TEST 3: SKIPPED - CONFIG_LIST not available")
    print()


# ============================================================================
# Test 4: Check what's inside the model files
# ============================================================================

print("TEST 4: Inspect model file structure")
print("-" * 80)

model_path = project_root / "github_models" / "ddqn_online_att_model_final.d3"

if model_path.exists():
    try:
        checkpoint = torch.load(str(model_path), map_location='cpu')
        print(f"Checkpoint type: {type(checkpoint)}")
        print(f"Top-level keys: {list(checkpoint.keys())}")
        print()

        # Look for encoder info
        if 'params' in checkpoint:
            print(f"Params keys: {list(checkpoint['params'].keys())}")
            print()

            # Look for encoder configuration
            for key in checkpoint['params'].keys():
                if 'encoder' in key.lower():
                    value = checkpoint['params'][key]
                    print(f"{key}:")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"  {value}")
                    print()

    except Exception as e:
        print(f"Failed to inspect model: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Model file not found: {model_path}")

print()
print("=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
