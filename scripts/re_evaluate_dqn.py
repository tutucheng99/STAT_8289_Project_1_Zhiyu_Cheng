"""
Re-evaluate DQN model with 500 episodes to match BC and CQL

This script loads the trained DQN model and re-evaluates it with 500 episodes
to ensure fair comparison across all algorithms.

Usage:
    python scripts/re_evaluate_dqn.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from stable_baselines3 import DQN

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env
from src.evaluation.metrics import evaluate_policy, print_evaluation_results


def main():
    """Re-evaluate DQN model with 500 episodes"""
    print("\n" + "="*60)
    print("RE-EVALUATING DQN MODEL")
    print("="*60 + "\n")

    n_episodes = 500

    # Load model
    model_path = project_root / "results" / "models" / "dqn_simple_reward.zip"

    if not model_path.exists():
        print("[ERROR] DQN model not found")
        return 1

    print("Loading DQN model...")
    dqn_model = DQN.load(str(model_path))
    print("[OK] DQN model loaded successfully")

    # Create environment
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)

    # Define policy function
    def dqn_policy(state):
        action, _ = dqn_model.predict(state, deterministic=True)
        return int(action)

    # Evaluate
    print(f"\nEvaluating DQN with {n_episodes} episodes...")
    print("(This will take ~5-8 minutes)")

    results = evaluate_policy(
        env=env,
        policy_fn=dqn_policy,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    # Print results
    print_evaluation_results(results, policy_name=f"DQN (re-evaluated, n={n_episodes})")

    # Update dqn_results.pkl
    print("\n" + "="*60)
    print("UPDATING DQN RESULTS FILE")
    print("="*60 + "\n")

    results_dir = project_root / "results"
    dqn_results_file = results_dir / "dqn_results.pkl"

    with open(dqn_results_file, 'rb') as f:
        dqn_data = pickle.load(f)

    # Update evaluation results
    dqn_data['evaluation'] = results
    dqn_data['evaluation_episodes'] = n_episodes

    with open(dqn_results_file, 'wb') as f:
        pickle.dump(dqn_data, f)

    print("[OK] Updated DQN results saved")

    # Load BC and CQL for comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON (ALL WITH 500 EPISODES)")
    print("="*60 + "\n")

    with open(results_dir / "bc_results.pkl", 'rb') as f:
        bc_data = pickle.load(f)
    bc_results = bc_data['evaluation']

    with open(results_dir / "cql_results.pkl", 'rb') as f:
        cql_data = pickle.load(f)
    cql_results = cql_data['evaluation']

    print(f"{'Algorithm':<15} {'Overall Survival':<20} {'High SOFA Survival':<25} {'High SOFA n':<15}")
    print("-" * 75)

    # BC
    bc_overall = bc_results['survival_rate'] * 100
    bc_high = bc_results['sofa_stratified']['high_sofa']['survival_rate'] * 100
    bc_high_n = bc_results['sofa_stratified']['high_sofa']['n_episodes']
    print(f"{'BC':<15} {bc_overall:>6.1f}%             {bc_high:>6.1f}%                  {bc_high_n:<15}")

    # CQL
    cql_overall = cql_results['survival_rate'] * 100
    cql_high = cql_results['sofa_stratified']['high_sofa']['survival_rate'] * 100
    cql_high_n = cql_results['sofa_stratified']['high_sofa']['n_episodes']
    print(f"{'CQL':<15} {cql_overall:>6.1f}%             {cql_high:>6.1f}%                  {cql_high_n:<15}")

    # DQN
    dqn_overall = results['survival_rate'] * 100
    dqn_high = results['sofa_stratified']['high_sofa']['survival_rate'] * 100
    dqn_high_n = results['sofa_stratified']['high_sofa']['n_episodes']
    print(f"{'DQN':<15} {dqn_overall:>6.1f}%             {dqn_high:>6.1f}%                  {dqn_high_n:<15}")

    print("\n" + "="*60)
    print("[OK] DQN RE-EVALUATION COMPLETE!")
    print("="*60)
    print("\nAll models now evaluated with 500 episodes for fair comparison.")
    print("High SOFA sample sizes: ~180-200 patients per algorithm")
    print("\nNext steps:")
    print("  1. Review the comparison table above")
    print("  2. Run visualization: python scripts/06_visualization.py")
    print("  3. Run final analysis: python scripts/07_final_analysis.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
