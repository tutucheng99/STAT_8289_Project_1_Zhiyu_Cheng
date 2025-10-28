"""
Main Experiment Runner

Runs all experiments in sequence or individually.

Usage:
    python run_experiments.py --all              # Run everything
    python run_experiments.py --baseline         # Step 1: Baseline evaluation
    python run_experiments.py --train-rl         # Step 2: Train RL algorithms
    python run_experiments.py --reward-comp      # Step 3: Reward comparison
    python run_experiments.py --visualize        # Step 4: Generate figures
    python run_experiments.py --analyze          # Step 5: Final analysis
"""

import sys
import argparse
import subprocess
from pathlib import Path
import time

project_root = Path(__file__).parent


def run_script(script_name, description):
    """Run a Python script and report results"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70 + "\n")

    script_path = project_root / "scripts" / script_name

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True,
            capture_output=False  # Show output in real-time
        )

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def run_baseline():
    """Step 1: Baseline Evaluation"""
    return run_script("01_baseline_evaluation.py", "Baseline Evaluation")


def run_train_rl():
    """Step 2: Train RL Algorithms"""
    print("\n" + "="*70)
    print("STEP 2: TRAINING RL ALGORITHMS")
    print("="*70)

    scripts = [
        ("02_train_bc.py", "BC (Behavior Cloning)"),
        ("03_train_cql.py", "CQL (Conservative Q-Learning)"),
        ("04_train_dqn.py", "DQN (Deep Q-Network)"),
    ]

    results = []
    for script, desc in scripts:
        success = run_script(script, desc)
        results.append((desc, success))

        if not success:
            print(f"\n⚠️  {desc} failed. Continue anyway? (y/n)")
            if input().lower() != 'y':
                return False

    # Summary
    print("\n" + "="*70)
    print("RL TRAINING SUMMARY")
    print("="*70)
    for desc, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {desc}")

    return all(success for _, success in results)


def run_reward_comparison():
    """Step 3: Reward Function Comparison"""
    return run_script("05_reward_comparison.py", "Reward Function Comparison")


def run_visualization():
    """Step 4: Generate Figures"""
    return run_script("06_visualization.py", "Visualization Generation")


def run_analysis():
    """Step 5: Final Analysis"""
    return run_script("07_final_analysis.py", "Final Analysis")


def run_all():
    """Run all experiments in sequence"""
    print("\n" + "="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)
    print("\nThis will take approximately 6-9 hours on CPU.")
    print("You can interrupt at any time with Ctrl+C.\n")

    steps = [
        ("Baseline Evaluation", run_baseline),
        ("RL Training", run_train_rl),
        ("Reward Comparison", run_reward_comparison),
        ("Visualization", run_visualization),
        ("Final Analysis", run_analysis),
    ]

    start_time = time.time()
    results = []

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"STEP {i}/{len(steps)}: {name}")
        print(f"{'='*70}")

        success = func()
        results.append((name, success))

        if not success:
            print(f"\n⚠️  {name} failed. Continue to next step? (y/n)")
            if input().lower() != 'y':
                break

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for name, success in results:
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")

    all_success = all(success for _, success in results)

    if all_success:
        print("\n🎉 All experiments completed successfully!")
        print("\nResults saved in: results/")
        print("  - Models: results/models/")
        print("  - Figures: results/figures/")
        print("  - Data: results/*.pkl")
        print("\nNext step: Write the paper!")
    else:
        print("\n⚠️  Some experiments failed. Check the output above.")

    return 0 if all_success else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run sepsis RL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --all          # Run all experiments
  python run_experiments.py --baseline     # Just baseline evaluation
  python run_experiments.py --train-rl     # Just train RL algorithms

Note: Experiments take ~6-9 hours total on CPU (perfectly fine!)
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Run all experiments in sequence')
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline evaluation')
    parser.add_argument('--train-rl', action='store_true',
                       help='Train RL algorithms (BC, CQL, DQN)')
    parser.add_argument('--reward-comp', action='store_true',
                       help='Reward function comparison')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate figures')
    parser.add_argument('--analyze', action='store_true',
                       help='Final analysis')

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 1

    # Run selected experiments
    if args.all:
        return run_all()

    if args.baseline:
        run_baseline()

    if args.train_rl:
        run_train_rl()

    if args.reward_comp:
        run_reward_comparison()

    if args.visualize:
        run_visualization()

    if args.analyze:
        run_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
