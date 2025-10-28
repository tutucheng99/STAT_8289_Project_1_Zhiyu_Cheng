"""
Create LEG Interpretability Comparison Figure

Compares the interpretability of BC, CQL, and DQN using LEG analysis results.
This figure demonstrates the key finding: CQL achieves comparable performance
with superior interpretability.

Usage:
    python scripts/create_leg_comparison_figure.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def create_leg_comparison_figure():
    """Create comprehensive LEG interpretability comparison"""

    # Data from LEG analysis results
    # CQL - Strong interpretability
    cql_data = {
        'State 1': {'SysBP': -40.06, 'MeanBP': -24.50, 'Glucose': 19.66, 'LACTATE': -12.23},
        'State 3': {'LACTATE': -37.75, 'SysBP': 9.36, 'Glucose': -5.98, 'MeanBP': -5.47},
        'State 8': {'LACTATE': -5.56, 'Glucose': -4.84, 'SysBP': 2.82, 'GLUCOSE': -1.96},
    }

    # BC - Mixed interpretability
    bc_data = {
        'State 5': {'SysBP': -0.78, 'qsofa_gcs_score': 0.25, 'Glucose': 0.21, 'MeanBP': -0.17},
        'State 1': {'blood_culture_positive': 0.00, 'HeartRate': 0.00, 'PT': 0.00, 'POTASSIUM': 0.00},
        'State 7': {'blood_culture_positive': 0.00, 'HeartRate': 0.00, 'PT': 0.00, 'POTASSIUM': 0.00},
    }

    # DQN - Weak interpretability
    dqn_data = {
        'State 1': {'INR': 0.069, 'Glucose': -0.049, 'qsofa_resprate_score': 0.046, 'ALBUMIN': 0.042},
        'State 2': {'BILIRUBIN': -0.061, 'LACTATE': -0.053, 'CHLORIDE': 0.041, 'RespRate': -0.037},
        'State 10': {'BILIRUBIN': -0.052, 'LACTATE': -0.044, 'Glucose': -0.033, 'elixhauser_hospital': -0.024},
    }

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, width_ratios=[2, 1])

    # Define colors
    colors = {'CQL': '#2ecc71', 'BC': '#f39c12', 'DQN': '#9b59b6'}

    # ============ Plot 1: Feature Importance Magnitude Comparison ============
    ax1 = fig.add_subplot(gs[0, :])

    algorithms = ['CQL', 'BC', 'DQN']
    max_saliencies = [40.06, 0.78, 0.069]
    typical_ranges = [
        [4, 40],    # CQL
        [0.05, 0.78],  # BC
        [0.02, 0.07]   # DQN
    ]

    x_pos = np.arange(len(algorithms))
    bars = ax1.bar(x_pos, max_saliencies, color=[colors[a] for a in algorithms], alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, max_saliencies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax1.set_ylabel('Maximum LEG Saliency Score', fontsize=14, fontweight='bold')
    ax1.set_title('LEG Interpretability Comparison: Feature Importance Magnitude',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms, fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotations
    ax1.annotate('600x stronger\nsignal', xy=(0, 40), xytext=(0.5, 20),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold', ha='center')

    ax1.annotate('Clinically\ninterpretable', xy=(0, 40), xytext=(0, 80),
                fontsize=11, color='green', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))

    ax1.annotate('Non-interpretable\n(uniform pattern)', xy=(2, 0.069), xytext=(2, 0.5),
                fontsize=11, color='darkred', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.5))

    # ============ Plot 2: CQL - High Interpretability ============
    ax2 = fig.add_subplot(gs[1, 0])

    # Collect all feature names and values for CQL
    all_features = []
    all_values = []
    all_states = []

    for state_name, features in cql_data.items():
        for feature, value in sorted(features.items(), key=lambda x: abs(x[1]), reverse=True):
            all_features.append(feature)
            all_values.append(value)
            all_states.append(state_name)

    y_pos = np.arange(len(all_features))
    bar_colors = ['#e74c3c' if v < 0 else '#3498db' for v in all_values]

    ax2.barh(y_pos, all_values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{f}\n({s})" for f, s in zip(all_features, all_states)], fontsize=9)
    ax2.set_xlabel('LEG Saliency Score', fontsize=12, fontweight='bold')
    ax2.set_title('CQL: Clinically Coherent Decision Rules', fontsize=13, fontweight='bold', color=colors['CQL'])
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    # Add text annotation
    ax2.text(0.02, 0.98, 'Blood pressure (SysBP, MeanBP) and\nlactate drive treatment decisions',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============ Plot 3: BC - Mixed Interpretability ============
    ax3 = fig.add_subplot(gs[1, 1])

    # Show contrast: one interpretable state vs one flat state
    interpretable_state = bc_data['State 5']
    flat_state = bc_data['State 1']

    features = list(interpretable_state.keys())
    values_interp = [interpretable_state[f] for f in features]
    values_flat = [flat_state.get(f, 0.0) for f in features]

    x = np.arange(len(features))
    width = 0.35

    ax3.barh(x - width/2, values_interp, width, label='State 5\n(interpretable)',
            color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax3.barh(x + width/2, values_flat, width, label='State 1\n(flat)',
            color='gray', alpha=0.5, edgecolor='black', linewidth=0.8)

    ax3.set_yticks(x)
    ax3.set_yticklabels(features, fontsize=9)
    ax3.set_xlabel('LEG Saliency Score', fontsize=12, fontweight='bold')
    ax3.set_title('BC: State-Dependent Pattern', fontsize=13, fontweight='bold', color=colors['BC'])
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.invert_yaxis()

    # ============ Plot 4: DQN - Weak Interpretability ============
    ax4 = fig.add_subplot(gs[2, 0])

    # Show uniformly weak importance across states
    all_features_dqn = []
    all_values_dqn = []
    all_states_dqn = []

    for state_name, features in dqn_data.items():
        for feature, value in sorted(features.items(), key=lambda x: abs(x[1]), reverse=True):
            all_features_dqn.append(feature)
            all_values_dqn.append(value)
            all_states_dqn.append(state_name)

    y_pos = np.arange(len(all_features_dqn))
    bar_colors_dqn = ['#e74c3c' if v < 0 else '#3498db' for v in all_values_dqn]

    ax4.barh(y_pos, all_values_dqn, color=bar_colors_dqn, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{f}\n({s})" for f, s in zip(all_features_dqn, all_states_dqn)], fontsize=9)
    ax4.set_xlabel('LEG Saliency Score', fontsize=12, fontweight='bold')
    ax4.set_title('DQN: Weak, Non-Interpretable Patterns', fontsize=13, fontweight='bold', color=colors['DQN'])
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.invert_yaxis()

    # Add text annotation
    ax4.text(0.02, 0.98, 'All features show weak, uniform\nimportance (max |saliency| < 0.07)',
            transform=ax4.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # ============ Plot 5: Summary Table ============
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    summary_data = [
        ['Algorithm', 'Max\nSaliency', 'Interp.\nRating', 'Clinical\nDeployment'],
        ['CQL', '40.06', 'Excellent', 'Suitable'],
        ['BC', '0.78', 'Mixed', 'Requires\nValidation'],
        ['DQN', '0.069', 'Poor', 'Limited']
    ]

    table = ax5.table(cellText=summary_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Color code algorithm rows
    for i, algo in enumerate(['CQL', 'BC', 'DQN'], start=1):
        cell = table[(i, 0)]
        cell.set_facecolor(colors[algo])
        cell.set_text_props(weight='bold', color='white')

    # Add title
    ax5.text(0.5, 0.95, 'Interpretability Summary\n(10 states per algorithm)',
            transform=ax5.transAxes, fontsize=12, fontweight='bold',
            ha='center', va='top')

    # ============ Overall Title ============
    fig.suptitle('LEG Interpretability Analysis: Performance-Interpretability Trade-off in Sepsis RL',
                fontsize=18, fontweight='bold', y=0.98)

    # Add footer
    fig.text(0.5, 0.01,
            'Key Finding: CQL achieves comparable survival rates (88.5% high SOFA) with superior interpretability (600x stronger feature signals)',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save figure
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "leg_interpretability_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("[OK] LEG comparison figure saved: results/figures/leg_interpretability_comparison.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("CREATING LEG INTERPRETABILITY COMPARISON FIGURE")
    print("="*60 + "\n")

    create_leg_comparison_figure()

    print("\n" + "="*60)
    print("FIGURE CREATION COMPLETE!")
    print("="*60)
    print("\nFigure saved:")
    print("  results/figures/leg_interpretability_comparison.png")
    print("\nThis figure demonstrates the key contribution:")
    print("  - CQL achieves comparable performance WITH high interpretability")
    print("  - 600x stronger feature importance signals vs DQN")
    print("  - Clinically coherent decision rules (blood pressure + lactate)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
