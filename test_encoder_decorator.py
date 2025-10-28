"""
Test register_encoder_factory as a decorator
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
from d3rlpy.models.encoders import Encoder, EncoderFactory, register_encoder_factory
from d3rlpy.types import Shape

print("=" * 80)
print("TESTING ENCODER REGISTRATION AS DECORATOR")
print("=" * 80)
print()

# ============================================================================
# Try using register_encoder_factory as a decorator
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


# Try Method 1: As decorator
print("Method 1: Using @register_encoder_factory as decorator")
print("-" * 80)
try:
    @register_encoder_factory
    class TestEncoderFactory1(EncoderFactory):
        """Factory for TestEncoder"""

        def create(self, observation_shape: Shape) -> TestEncoder:
            return TestEncoder(observation_shape)

    print("[OK] Decorator syntax succeeded!")
    print(f"Factory type: {type(TestEncoderFactory1)}")
    print()
except Exception as e:
    print(f"[ERROR] Decorator syntax failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# Try Method 2: Manual call with factory only
print("Method 2: Calling register_encoder_factory(factory_class)")
print("-" * 80)
try:
    class TestEncoderFactory2(EncoderFactory):
        """Factory for TestEncoder"""

        def create(self, observation_shape: Shape) -> TestEncoder:
            return TestEncoder(observation_shape)

    result = register_encoder_factory(TestEncoderFactory2)
    print(f"[OK] Manual call succeeded!")
    print(f"Result type: {type(result)}")
    print()
except Exception as e:
    print(f"[ERROR] Manual call failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# Try Method 3: Inspect what register_encoder_factory actually is
print("Method 3: Inspecting register_encoder_factory")
print("-" * 80)
print(f"Type: {type(register_encoder_factory)}")
print(f"Module: {register_encoder_factory.__module__}")

import inspect
sig = inspect.signature(register_encoder_factory)
print(f"Signature: {sig}")
print()

# Get docstring
if register_encoder_factory.__doc__:
    print("Documentation:")
    print(register_encoder_factory.__doc__)
else:
    print("No documentation available")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
