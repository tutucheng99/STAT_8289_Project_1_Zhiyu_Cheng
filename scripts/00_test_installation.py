"""
Test Installation Script

Verifies that all dependencies are correctly installed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    print("="*60)

    try:
        import numpy as np
        print(f"[OK] NumPy: {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy: {e}")
        return False

    try:
        import pandas as pd
        print(f"[OK] Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pandas: {e}")
        return False

    try:
        import matplotlib
        print(f"[OK] Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"[FAIL] Matplotlib: {e}")
        return False

    try:
        import gym
        print(f"[OK] Gym: {gym.__version__}")
    except ImportError as e:
        print(f"[FAIL] Gym: {e}")
        return False

    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"[FAIL] TensorFlow: {e}")
        return False

    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"[FAIL] PyTorch: {e}")
        return False

    try:
        import d3rlpy
        print(f"[OK] d3rlpy: {d3rlpy.__version__}")
    except ImportError as e:
        print(f"[FAIL] d3rlpy: {e}")
        return False

    try:
        from stable_baselines3 import DQN
        print(f"[OK] Stable-Baselines3: installed")
    except ImportError as e:
        print(f"[FAIL] Stable-Baselines3: {e}")
        return False

    print("="*60)
    return True


def test_gym_sepsis():
    """Test gym-sepsis installation"""
    print("\nTesting gym-sepsis...")
    print("="*60)

    try:
        from gym_sepsis.envs.sepsis_env import SepsisEnv
        print("[OK] gym-sepsis: imported")

        # Try to create environment
        env = SepsisEnv(verbose=False)
        print("[OK] SepsisEnv: created successfully")

        # Test reset
        obs, info = env.reset()
        print(f"[OK] Environment reset: obs shape = {obs.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[OK] Environment step: reward = {reward:.2f}")

        env.close()
        print("[OK] gym-sepsis: fully functional")

    except Exception as e:
        print(f"[FAIL] gym-sepsis: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("="*60)
    return True


def test_custom_modules():
    """Test custom project modules"""
    print("\nTesting custom modules...")
    print("="*60)

    try:
        from src.envs.reward_functions import simple_reward, paper_reward, hybrid_reward
        print("[OK] Reward functions: imported")
    except ImportError as e:
        print(f"[FAIL] Reward functions: {e}")
        return False

    try:
        from src.envs.sepsis_wrapper import make_sepsis_env
        print("[OK] Environment wrapper: imported")

        # Test wrapper
        env = make_sepsis_env(reward_fn_name='simple', verbose=False)
        obs, info = env.reset()
        print(f"[OK] Wrapper functional: obs shape = {obs.shape}")
        env.close()

    except Exception as e:
        print(f"[FAIL] Environment wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from src.evaluation.metrics import evaluate_policy
        print("[OK] Evaluation metrics: imported")
    except ImportError as e:
        print(f"[FAIL] Evaluation metrics: {e}")
        return False

    try:
        from src.visualization.policy_viz import create_policy_heatmap
        print("[OK] Visualization tools: imported")
    except ImportError as e:
        print(f"[FAIL] Visualization tools: {e}")
        return False

    print("="*60)
    return True


def test_data_files():
    """Test that required data files exist"""
    print("\nChecking data files...")
    print("="*60)

    data_file = project_root / "data" / "offline_dataset.pkl"

    if data_file.exists():
        print(f"[OK] Offline dataset: {data_file}")

        # Try to load it
        try:
            import pickle
            with open(data_file, 'rb') as f:
                dataset = pickle.load(f)
            print(f"[OK] Dataset loaded: {len(dataset['observations'])} transitions")
        except Exception as e:
            print(f"[WARN] Dataset exists but cannot load: {e}")
    else:
        print(f"[WARN] Offline dataset not found: {data_file}")
        print("   This is optional - you can generate it later")

    print("="*60)
    return True


def test_directories():
    """Check/create necessary directories"""
    print("\nChecking directories...")
    print("="*60)

    dirs = [
        project_root / "results",
        project_root / "results" / "models",
        project_root / "results" / "figures",
        project_root / "results" / "logs",
    ]

    for d in dirs:
        if d.exists():
            print(f"[OK] Directory exists: {d.name}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Directory created: {d.name}")

    print("="*60)
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("INSTALLATION TEST")
    print("="*60 + "\n")

    tests = [
        ("Core packages", test_imports),
        ("gym-sepsis", test_gym_sepsis),
        ("Custom modules", test_custom_modules),
        ("Data files", test_data_files),
        ("Directories", test_directories),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\nAll tests passed! Ready to run experiments.")
        print("\nNext step:")
        print("  python run_experiments.py --baseline")
        return 0
    else:
        print("\nSome tests failed. Please fix before running experiments.")
        print("\nCommon fixes:")
        print("  pip install -r requirements.txt")
        print("  pip install -e gym-sepsis")
        return 1


if __name__ == "__main__":
    sys.exit(main())
