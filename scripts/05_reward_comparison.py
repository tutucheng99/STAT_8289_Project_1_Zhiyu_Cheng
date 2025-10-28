"""
Script 5: Reward Function Comparison

Compares different reward functions using the best performing algorithm.
Trains with 3 reward variants: simple, paper, hybrid.

Usage:
    python scripts/05_reward_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env
from src.evaluation.metrics import evaluate_policy, print_evaluation_results, compare_policies

# Import training functions from previous scripts
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset
from stable_baselines3 import DQN


def load_offline_dataset():
    """Load offline dataset"""
    data_file = project_root / "data" / "offline_dataset.pkl"

    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    observations = np.array(dataset['observations'], dtype=np.float32)
    actions = np.array(dataset['actions'], dtype=np.int32)
    rewards = np.array(dataset['rewards'], dtype=np.float32)
    terminals = np.array(dataset['terminals'], dtype=np.float32)

    mdp_dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )

    return mdp_dataset


def select_best_algorithm():
    """Select the best performing algorithm from Phase 1"""
    print("\n" + "="*60)
    print("SELECTING BEST ALGORITHM")
    print("="*60 + "\n")

    results_dir = project_root / "results"

    # Load all results
    algorithms = {}

    # Load baseline
    baseline_file = results_dir / "baseline_results.pkl"
    if baseline_file.exists():
        with open(baseline_file, 'rb') as f:
            baseline = pickle.load(f)
        algorithms['Random'] = baseline['random']
        algorithms['Heuristic'] = baseline['heuristic']

    # Load RL results
    for algo_name in ['bc', 'cql', 'dqn']:
        result_file = results_dir / f"{algo_name}_results.pkl"
        if result_file.exists():
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            algorithms[algo_name.upper()] = result['evaluation']

    # Print comparison
    print("Algorithm Performance (simple reward):\n")
    print(f"{'Algorithm':<15} {'Survival Rate':<15} {'Avg Return':<15}")
    print("-" * 60)

    best_algo = None
    best_survival = -1

    for name, results in algorithms.items():
        survival = results['survival_rate'] * 100
        avg_return = results['avg_return']

        print(f"{name:<15} {survival:>6.1f}%          {avg_return:>6.2f}")

        # Track best (prioritize RL methods over baselines)
        if name not in ['Random', 'Heuristic'] and survival > best_survival:
            best_survival = survival
            best_algo = name

    print("\n" + "="*60)
    print(f"SELECTED: {best_algo} (Survival: {best_survival:.1f}%)")
    print("="*60)

    return best_algo


def train_with_reward(algorithm, reward_fn_name, mdp_dataset=None):
    """Train algorithm with specific reward function"""
    print(f"\n{'='*60}")
    print(f"TRAINING {algorithm.upper()} WITH {reward_fn_name.upper()} REWARD")
    print(f"{'='*60}\n")

    if algorithm == 'CQL':
        # Train CQL (offline)
        print("Training CQL (offline)...")
        cql = DiscreteCQLConfig(
            batch_size=256,
            learning_rate=3e-4,
            target_update_interval=2000,
            alpha=1.0,
        ).create(device='cpu')

        cql.fit(mdp_dataset, n_epochs=100, show_progress=True)

        # Evaluate
        env = make_sepsis_env(reward_fn_name=reward_fn_name, verbose=False)

        def policy_fn(state):
            state_batch = np.array([state], dtype=np.float32)
            action = cql.predict(state_batch)[0]
            return int(action)

        # ⚡ INCREASED: 500 episodes for more reliable estimates
        results = evaluate_policy(env, policy_fn, n_episodes=500, max_steps=50, verbose=True)
        env.close()

        # Save model
        models_dir = project_root / "results" / "models"
        model_path = models_dir / f"cql_{reward_fn_name}_reward.d3"
        cql.save(str(model_path))

        return results, str(model_path)

    elif algorithm == 'DQN':
        # Train DQN (online) - OPTIMIZED VERSION
        print("Training DQN (online) with optimized parameters...")
        env = make_sepsis_env(reward_fn_name=reward_fn_name, verbose=False)

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=20000,          # Reduced: episodes are short (~10 steps)
            learning_starts=2000,       # Start learning after 2000 steps
            batch_size=1024,            # Increased: 4x larger for efficiency
            tau=0.005,
            gamma=0.99,
            train_freq=8,               # Train every 8 steps (less frequent = faster)
            target_update_interval=500, # Update target more frequently
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=0,
            device='cpu'
        )

        # ⚡ OPTIMIZED: 50K timesteps (vs 200K original)
        # Estimated time: 20-30 min per reward function
        model.learn(total_timesteps=50000, progress_bar=True)

        # Evaluate
        def policy_fn(state):
            action, _ = model.predict(state, deterministic=True)
            return int(action)

        # ⚡ INCREASED: 500 episodes for more reliable estimates (especially for high-SOFA patients)
        # This reduces standard error from ±3.5% to ±2.2% for high-SOFA subgroup
        results = evaluate_policy(env, policy_fn, n_episodes=500, max_steps=50, verbose=True)
        env.close()

        # Save model
        models_dir = project_root / "results" / "models"
        model_path = models_dir / f"dqn_{reward_fn_name}_reward.zip"
        model.save(str(model_path))

        return results, str(model_path)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    """Run reward function comparison"""
    print("\n" + "="*60)
    print("REWARD FUNCTION COMPARISON")
    print("="*60 + "\n")

    # Select best algorithm
    best_algo = select_best_algorithm()

    # Load offline dataset (for CQL)
    mdp_dataset = None
    if best_algo == 'CQL':
        print("\nLoading offline dataset...")
        mdp_dataset = load_offline_dataset()

    # Train with all 3 reward functions
    reward_functions = ['simple', 'paper', 'hybrid']
    all_results = {}

    for reward_fn in reward_functions:
        # ⚡ OPTIMIZATION: Skip training if model already exists
        models_dir = project_root / "results" / "models"
        if best_algo == 'DQN':
            existing_model = models_dir / f"dqn_{reward_fn}_reward.zip"
        elif best_algo == 'CQL':
            existing_model = models_dir / f"cql_{reward_fn}_reward.d3"
        else:
            existing_model = models_dir / f"{best_algo.lower()}_{reward_fn}_reward.d3"

        if existing_model.exists():
            print(f"\n⚡ SKIPPING {reward_fn.upper()} reward - model already exists!")
            print(f"   Loading existing model: {existing_model}")

            # Load and evaluate existing model
            env = make_sepsis_env(reward_fn_name=reward_fn, verbose=False)

            if best_algo == 'DQN':
                from stable_baselines3 import DQN
                model = DQN.load(str(existing_model))
                def policy_fn(state):
                    action, _ = model.predict(state, deterministic=True)
                    return int(action)
            elif best_algo == 'CQL':
                import d3rlpy
                model = d3rlpy.load_learnable(str(existing_model))
                def policy_fn(state):
                    state_batch = np.array([state], dtype=np.float32)
                    action = model.predict(state_batch)[0]
                    return int(action)

            # ⚡ INCREASED: 500 episodes for more reliable estimates
            results = evaluate_policy(env, policy_fn, n_episodes=500, max_steps=50, verbose=True)
            env.close()
            all_results[reward_fn] = results

            print(f"\n✅ {reward_fn.upper()} reward results (existing model):")
            print_evaluation_results(results, policy_name=f"{best_algo} ({reward_fn})")
            continue

        # Train new model
        results, model_path = train_with_reward(best_algo, reward_fn, mdp_dataset)
        all_results[reward_fn] = results

        print(f"\n✅ {reward_fn.upper()} reward results:")
        print_evaluation_results(results, policy_name=f"{best_algo} ({reward_fn})")

    # Compare results
    print("\n" + "="*60)
    print("REWARD FUNCTION COMPARISON")
    print("="*60 + "\n")

    print(f"{'Reward Function':<20} {'Survival Rate':<15} {'Avg Return':<15}")
    print("-" * 60)

    for reward_fn, results in all_results.items():
        survival = results['survival_rate'] * 100
        avg_return = results['avg_return']
        print(f"{reward_fn.capitalize():<20} {survival:>6.1f}%          {avg_return:>6.2f}")

    # Save results
    results_dir = project_root / "results"
    results_file = results_dir / "reward_comparison_results.pkl"

    comparison_results = {
        'algorithm': best_algo,
        'results': all_results
    }

    with open(results_file, 'wb') as f:
        pickle.dump(comparison_results, f)
    print(f"\n✅ Results saved: {results_file}")

    # Visualization
    print("\nGenerating comparison figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Survival Rate
    reward_names = [r.capitalize() for r in reward_functions]
    survival_rates = [all_results[r]['survival_rate'] * 100 for r in reward_functions]

    axes[0].bar(reward_names, survival_rates, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Survival Rate (%)')
    axes[0].set_title(f'{best_algo}: Survival Rate by Reward Function')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(survival_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Plot 2: Average Return
    avg_returns = [all_results[r]['avg_return'] for r in reward_functions]
    std_returns = [all_results[r]['std_return'] for r in reward_functions]

    axes[1].bar(reward_names, avg_returns, yerr=std_returns,
                color=['#3498db', '#e74c3c', '#2ecc71'], capsize=5)
    axes[1].set_ylabel('Average Return')
    axes[1].set_title(f'{best_algo}: Average Return by Reward Function')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figure
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "reward_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved: {fig_path}")
    plt.close()

    # Summary
    print("\n" + "="*60)
    print("REWARD COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nBest Algorithm: {best_algo}")
    print(f"\nResults across reward functions:")
    for reward_fn in reward_functions:
        survival = all_results[reward_fn]['survival_rate'] * 100
        print(f"  {reward_fn.capitalize()}: {survival:.1f}% survival")
    print(f"\nNext step:")
    print(f"  python scripts/06_visualization.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
