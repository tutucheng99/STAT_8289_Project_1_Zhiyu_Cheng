"""
DQN Policy Interpretability Analysis

Demonstrates how to interpret and explain DQN policy decisions.
Run this after training a DQN model to understand what it learned.

Usage:
    python scripts/interpret_dqn.py --model results/models/dqn_simple_reward.zip --n_examples 5
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env
from src.visualization.interpretability import (
    analyze_q_values,
    feature_importance_simple,
    compare_with_clinician,
    explain_single_decision,
    plot_q_value_landscape,
    plot_feature_importance
)

# Import Stable-Baselines3 for loading DQN models
from stable_baselines3 import DQN


def load_dqn_model(model_path: str):
    """Load trained Stable-Baselines3 DQN model"""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load DQN model from .zip file
    if model_path.suffix.lower() == '.zip':
        model = DQN.load(str(model_path), device='cpu')
        print(f"[OK] Loaded DQN model from {model_path}")
        return model
    else:
        raise ValueError(f"Expected .zip file for DQN model, got {model_path.suffix}")


class DQNWrapper:
    """
    Wrapper to make Stable-Baselines3 DQN compatible with interpretability functions
    that expect d3rlpy-style interface
    """
    def __init__(self, sb3_model):
        self.sb3_model = sb3_model
        self.n_actions = sb3_model.action_space.n

    def predict(self, obs):
        """
        Predict action for given observation(s)
        Returns: numpy array of actions matching d3rlpy interface
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action, _states = self.sb3_model.predict(obs, deterministic=True)

        # Always return as numpy array for compatibility
        if not isinstance(action, np.ndarray):
            action = np.array([action])

        return action

    def predict_value(self, obs, action=None):
        """
        Get Q-values for observation(s)
        If action is None, returns Q-values for all actions
        If action is specified, returns Q-value for that action
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Get Q-values from DQN's Q-network
        import torch
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.sb3_model.device)
            q_values = self.sb3_model.q_net(obs_tensor).cpu().numpy()

        if action is None:
            return q_values
        else:
            if isinstance(action, (list, np.ndarray)):
                return q_values[np.arange(len(action)), action]
            else:
                return q_values[0, action]


def heuristic_policy(state):
    """Simple clinician heuristic for comparison"""
    LACTATE_IDX = 15
    MEAN_BP_IDX = 16
    SBP_IDX = 25
    SOFA_IDX = 37

    lactate = state[LACTATE_IDX]
    sbp = state[SBP_IDX]
    map_bp = state[MEAN_BP_IDX]
    sofa = state[SOFA_IDX]

    # Clinical rules
    if sbp < -1.0 or map_bp < -1.0:
        iv_bin, vp_bin = 4, 3
    elif lactate > 1.0:
        iv_bin, vp_bin = 3, 2
    elif sofa > 1.0:
        iv_bin, vp_bin = 3, 3
    elif sbp < 0 or lactate > 0:
        iv_bin, vp_bin = 2, 1
    else:
        iv_bin, vp_bin = 1, 1

    return min(5 * iv_bin + vp_bin, 23)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='results/models/dqn_simple_reward.zip',
                        help='Path to trained DQN model (.zip file)')
    parser.add_argument('--n_examples', type=int, default=5,
                        help='Number of example cases to analyze')
    args = parser.parse_args()

    print("=" * 60)
    print("DQN POLICY INTERPRETABILITY ANALYSIS")
    print("=" * 60 + "\n")

    # Load DQN model
    sb3_dqn = load_dqn_model(args.model)

    # Wrap it for compatibility with interpretability functions
    model = DQNWrapper(sb3_dqn)

    # Extract model name for file naming
    model_path = Path(args.model)
    model_name = model_path.stem  # e.g., "dqn_simple_reward"
    print(f"Model identifier: {model_name}\n")

    # Create environment
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)
    print(f"[OK] Environment created\n")

    # Create policy function
    def rl_policy(state):
        return int(model.predict(state.reshape(1, -1))[0])

    print("-" * 60)
    print("1. CLINICIAN AGREEMENT ANALYSIS")
    print("-" * 60)

    comparison = compare_with_clinician(
        rl_policy=rl_policy,
        clinician_policy=heuristic_policy,
        env=env,
        n_episodes=100
    )

    print(f"\nAgreement Rate: {comparison['overall_agreement_rate']*100:.1f}%")
    print(f"Std: {comparison['agreement_std']*100:.1f}%")
    print(f"Total Disagreements: {comparison['n_disagreements']}")

    if comparison['disagreement_details']:
        print(f"\nExample Disagreement Cases:")
        for i, case in enumerate(comparison['disagreement_details'][:3]):
            print(f"\n  Case {i+1}:")
            print(f"    SOFA: {case['sofa']:.2f}")
            print(f"    Lactate: {case['lactate']:.2f}")
            print(f"    Mean BP: {case['map_bp']:.2f}")
            print(f"    DQN Action: {case['rl_action']}")
            print(f"    Clinician Action: {case['clinician_action']}")

    print("\n" + "-" * 60)
    print("2. DETAILED CASE EXAMPLES")
    print("-" * 60)

    # Analyze specific cases
    figures_dir = project_root / "results" / "figures" / "interpretability"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_examples):
        print(f"\n{'='*60}")
        print(f"Case {i+1}")
        print(f"{'='*60}")

        # Reset and get a state
        obs, info = env.reset()

        # Explain decision
        explanation = explain_single_decision(model, obs)
        print(explanation)

        # Feature importance
        print("Computing feature importance...")
        importance_df = feature_importance_simple(model, obs)

        print("\nTop 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<25} Importance: {row['importance']:.3f}  Value: {row['value']:.2f}")

        # Visualizations
        print("\nGenerating visualizations...")

        # Q-value landscape
        fig1 = plot_q_value_landscape(
            model, obs,
            save_path=figures_dir / f"{model_name}_q_landscape_case_{i+1}.png"
        )
        print(f"  [OK] Saved Q-value landscape")

        # Feature importance plot
        fig2 = plot_feature_importance(
            importance_df,
            save_path=figures_dir / f"{model_name}_feature_importance_case_{i+1}.png"
        )
        print(f"  [OK] Saved feature importance plot")

        import matplotlib.pyplot as plt
        plt.close('all')

    env.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {figures_dir}")
    print(f"\nFile naming pattern:")
    print(f"  - Q-value landscapes: {model_name}_q_landscape_case_*.png")
    print(f"  - Feature importance: {model_name}_feature_importance_case_*.png")
    print("\nKey Insights:")
    print(f"  - DQN policy agrees with clinician {comparison['overall_agreement_rate']*100:.1f}% of the time")
    print(f"  - Analyzed {args.n_examples} detailed cases")
    print(f"  - Generated {args.n_examples * 2} plots (Q-values + feature importance)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
