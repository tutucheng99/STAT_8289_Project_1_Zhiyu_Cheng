"""
Re-evaluate BC and CQL models with 500 episodes for fair comparison with DQN

This script loads the trained BC and CQL models and re-evaluates them with 500 episodes
to match the DQN evaluation protocol.

Usage:
    python scripts/re_evaluate_all.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import d3rlpy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env
from src.evaluation.metrics import evaluate_policy, print_evaluation_results


def evaluate_bc_model(n_episodes=500):
    """Re-evaluate BC model with specified number of episodes"""
    print("\n" + "="*60)
    print("RE-EVALUATING BC MODEL")
    print("="*60 + "\n")

    # Load model
    model_path = project_root / "results" / "models" / "bc_simple_reward.d3"

    if not model_path.exists():
        print("[ERROR] BC model not found")
        return None

    print("Loading BC model...")
    bc_model = d3rlpy.load_learnable(str(model_path))
    print("[OK] BC model loaded successfully")

    # Create environment
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)

    # Define policy function
    def bc_policy(state):
        state_batch = np.array([state], dtype=np.float32)
        action = bc_model.predict(state_batch)[0]
        return int(action)

    # Evaluate
    print(f"\nEvaluating BC with {n_episodes} episodes...")
    print("(This will take ~5-8 minutes)")

    results = evaluate_policy(
        env=env,
        policy_fn=bc_policy,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    # Print results
    print_evaluation_results(results, policy_name=f"BC (re-evaluated, n={n_episodes})")

    return results


def evaluate_cql_model(n_episodes=500):
    """Re-evaluate CQL model with specified number of episodes"""
    print("\n" + "="*60)
    print("RE-EVALUATING CQL MODEL")
    print("="*60 + "\n")

    # Load model
    model_path = project_root / "results" / "models" / "cql_simple_reward.d3"

    if not model_path.exists():
        print("[ERROR] CQL model not found")
        return None

    print("Loading CQL model...")
    cql_model = d3rlpy.load_learnable(str(model_path))
    print("[OK] CQL model loaded successfully")

    # Create environment
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)

    # Define policy function
    def cql_policy(state):
        state_batch = np.array([state], dtype=np.float32)
        action = cql_model.predict(state_batch)[0]
        return int(action)

    # Evaluate
    print(f"\nEvaluating CQL with {n_episodes} episodes...")
    print("(This will take ~5-8 minutes)")

    results = evaluate_policy(
        env=env,
        policy_fn=cql_policy,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    # Print results
    print_evaluation_results(results, policy_name=f"CQL (re-evaluated, n={n_episodes})")

    return results


def main():
    """Re-evaluate both BC and CQL models"""
    print("\n" + "="*60)
    print("RE-EVALUATION FOR FAIR COMPARISON")
    print("="*60)
    print("\nPurpose: Evaluate BC and CQL with 500 episodes to match DQN")
    print("Reason: Original evaluations used 200 episodes, causing unfair comparison")
    print("Expected time: ~10-15 minutes total\n")

    n_episodes = 500

    # Re-evaluate BC
    print("\n" + "="*60)
    print("STEP 1/2: BC EVALUATION")
    print("="*60)

    bc_results = evaluate_bc_model(n_episodes=n_episodes)

    if bc_results is None:
        print("\n[ERROR] BC evaluation failed. Exiting.")
        return 1

    # Re-evaluate CQL
    print("\n" + "="*60)
    print("STEP 2/2: CQL EVALUATION")
    print("="*60)

    cql_results = evaluate_cql_model(n_episodes=n_episodes)

    if cql_results is None:
        print("\n[ERROR] CQL evaluation failed. Exiting.")
        return 1

    # Save updated results
    print("\n" + "="*60)
    print("SAVING UPDATED RESULTS")
    print("="*60 + "\n")

    results_dir = project_root / "results"

    # Update BC results
    bc_results_file = results_dir / "bc_results.pkl"
    with open(bc_results_file, 'rb') as f:
        bc_data = pickle.load(f)

    bc_data['evaluation'] = bc_results
    bc_data['evaluation_episodes'] = n_episodes

    with open(bc_results_file, 'wb') as f:
        pickle.dump(bc_data, f)
    print("[OK] Updated BC results saved")

    # Update CQL results
    cql_results_file = results_dir / "cql_results.pkl"
    with open(cql_results_file, 'rb') as f:
        cql_data = pickle.load(f)

    cql_data['evaluation'] = cql_results
    cql_data['evaluation_episodes'] = n_episodes

    with open(cql_results_file, 'wb') as f:
        pickle.dump(cql_data, f)
    print("[OK] Updated CQL results saved")

    # Create comparison summary
    print("\n" + "="*60)
    print("FINAL COMPARISON (ALL WITH 500 EPISODES)")
    print("="*60 + "\n")

    # Load DQN results (should already be 500 episodes from reward comparison)
    dqn_results_file = results_dir / "dqn_results.pkl"
    with open(dqn_results_file, 'rb') as f:
        dqn_data = pickle.load(f)
    dqn_results = dqn_data['evaluation']

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
    dqn_overall = dqn_results['survival_rate'] * 100
    dqn_high = dqn_results['sofa_stratified']['high_sofa']['survival_rate'] * 100
    dqn_high_n = dqn_results['sofa_stratified']['high_sofa']['n_episodes']
    print(f"{'DQN':<15} {dqn_overall:>6.1f}%             {dqn_high:>6.1f}%                  {dqn_high_n:<15}")

    print("\n" + "="*60)
    print("[OK] RE-EVALUATION COMPLETE!")
    print("="*60)
    print("\nAll models now evaluated with 500 episodes for fair comparison.")
    print("High SOFA sample sizes: ~180-200 patients per algorithm")
    print("\nNext steps:")
    print("  1. Review the comparison table above")
    print("  2. Determine final narrative for paper")
    print("  3. Update paper/DATA_ANALYSIS.md with new findings")
    print("  4. Proceed with paper writing based on reliable data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
