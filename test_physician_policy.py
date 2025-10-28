"""
Test if gym-sepsis has built-in physician policy

This script checks:
1. Whether gym-sepsis environment has built-in physician policy
2. Test the heuristic_policy (clinical rule-based policy) in the project
3. Show how to use these policies
"""

import sys
from pathlib import Path
import numpy as np
import inspect

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "gym-sepsis"))
sys.path.insert(0, str(project_root / "scripts"))

print("="*80)
print("Testing if gym-sepsis has built-in physician policy")
print("="*80)

# Define heuristic_policy locally (from scripts/01_baseline_evaluation.py)
def heuristic_policy(state):
    """
    Clinical rule-based heuristic policy

    Decision rules based on SOFA, lactate, blood pressure
    """
    # Feature indices
    LACTATE_IDX = 15
    MEAN_BP_IDX = 16
    SBP_IDX = 25
    SOFA_IDX = 37

    lactate = state[LACTATE_IDX]
    sbp = state[SBP_IDX]
    map_bp = state[MEAN_BP_IDX]
    sofa = state[SOFA_IDX]

    # Clinical decision rules (states are standardized)
    if sbp < -1.0 or map_bp < -1.0:  # Severe hypotension
        iv_bin, vp_bin = 4, 3
    elif lactate > 1.0:  # High lactate
        iv_bin, vp_bin = 3, 2
    elif sofa > 1.0:  # High SOFA
        iv_bin, vp_bin = 3, 3
    elif sbp < 0 or lactate > 0:  # Mild abnormalities
        iv_bin, vp_bin = 2, 1
    else:  # Stable
        iv_bin, vp_bin = 1, 1

    action = min(5 * iv_bin + vp_bin, 23)
    return action

# ============================================================
# 1. Check gym-sepsis environment
# ============================================================
print("\n[1] Checking gym-sepsis environment...")
print("-"*80)

try:
    from gym_sepsis.envs.sepsis_env import SepsisEnv

    # Check all methods and attributes of SepsisEnv class
    sepsis_methods = [method for method in dir(SepsisEnv) if not method.startswith('_')]

    print(f"SepsisEnv public methods and attributes:")
    for method in sepsis_methods:
        print(f"  - {method}")

    # Check if there are physician/doctor/clinician/expert related methods
    physician_related = [m for m in sepsis_methods if any(
        keyword in m.lower() for keyword in ['physician', 'doctor', 'clinician', 'expert', 'policy']
    )]

    if physician_related:
        print(f"\nFound physician-related methods: {physician_related}")
    else:
        print(f"\nX gym-sepsis does NOT have built-in physician policy methods")

except Exception as e:
    print(f"Error: {e}")


# ============================================================
# 2. Check heuristic_policy in the project
# ============================================================
print("\n[2] Checking heuristic_policy (clinical rule-based policy)...")
print("-"*80)

try:
    # Show policy source code
    print("Heuristic Policy Source Code:")
    print(inspect.getsource(heuristic_policy))

    print("\nOK Project has heuristic_policy (clinical rule-based heuristic policy)")
    print("  This policy simulates clinical decision-making based on:")
    print("  - Blood Pressure (SysBP, MeanBP)")
    print("  - Lactate")
    print("  - SOFA score")

except Exception as e:
    print(f"Error: {e}")


# ============================================================
# 3. Test heuristic_policy
# ============================================================
print("\n[3] Testing heuristic_policy...")
print("-"*80)

try:
    from src.envs.sepsis_wrapper import make_sepsis_env

    # Create environment
    env = make_sepsis_env(reward_fn_name='simple')
    print("OK Environment created successfully")

    # Run several episodes
    print("\nRunning 5 test episodes:")

    for ep in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 50:
            # Use heuristic_policy to select action
            action = heuristic_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

        survived = "Survived" if total_reward > 0 else "Died"
        print(f"  Episode {ep+1}: steps={steps:2d}, total_reward={total_reward:6.1f}, outcome={survived}")

    env.close()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# 4. Compare actions under different states
# ============================================================
print("\n[4] Demonstrating action selection under different patient states...")
print("-"*80)

try:
    # Create test states (simplified, 46-dim vectors)
    print("\nHeuristic Policy decision rules:")
    print("  Feature indices: LACTATE=15, MeanBP=16, SysBP=25, SOFA=37")
    print("  Action encoding: action = min(5*iv_bin + vp_bin, 23)")
    print("  iv_bin: IV fluid dose bin (0-4)")
    print("  vp_bin: Vasopressor dose bin (0-4)")

    # Simulate different clinical scenarios
    scenarios = [
        {
            "name": "Stable Patient",
            "lactate": -0.5,  # standardized (below mean)
            "sbp": 0.5,       # standardized (above mean)
            "map_bp": 0.3,
            "sofa": -0.6
        },
        {
            "name": "Mild Abnormal",
            "lactate": 0.5,   # standardized (slightly high)
            "sbp": -0.3,
            "map_bp": -0.2,
            "sofa": 0.2
        },
        {
            "name": "High Lactate",
            "lactate": 1.5,   # standardized (high)
            "sbp": 0.0,
            "map_bp": 0.0,
            "sofa": 0.5
        },
        {
            "name": "Severe Hypotension",
            "lactate": 0.3,
            "sbp": -1.5,      # standardized (severely low)
            "map_bp": -1.2,
            "sofa": 1.0
        }
    ]

    print("\nTreatment decisions under different clinical scenarios:\n")

    for scenario in scenarios:
        # Create a 46-dim state vector (mostly zeros)
        state = np.zeros(46)
        state[15] = scenario["lactate"]
        state[16] = scenario["map_bp"]
        state[25] = scenario["sbp"]
        state[37] = scenario["sofa"]

        # Get action
        action = heuristic_policy(state)
        iv_dose = action // 5
        vp_dose = action % 5

        print(f"{scenario['name']:20s} | Lactate={scenario['lactate']:5.1f}, "
              f"SBP={scenario['sbp']:5.1f}, MAP={scenario['map_bp']:5.1f}, "
              f"SOFA={scenario['sofa']:5.1f} -> "
              f"Action={action:2d} (IV={iv_dose}, VP={vp_dose})")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
1. X gym-sepsis environment does NOT have a built-in "physician policy"

2. OK But the project implements heuristic_policy (heuristic policy), which is:
   - A clinical rule-based policy
   - Simulates clinical decision-making logic
   - Makes decisions based on blood pressure, lactate, SOFA score, etc.

3. In RL literature, this type of policy is often called:
   - Heuristic Policy
   - Clinician Policy
   - Behavioral Policy
   - Expert Policy

4. This policy is used for:
   - Collecting offline datasets (for training BC and CQL)
   - Serving as a baseline for comparison
   - Interpretability analysis (comparing RL policy with clinical rules)

If you want to add a real "physician policy" (based on actual clinical data):
- Extract real physician treatment decisions from MIMIC-III database
- Train a Behavior Cloning model to imitate physicians
- Or implement more complex clinical guideline rules
""")
