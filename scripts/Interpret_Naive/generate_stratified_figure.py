"""
Generate stratified survival comparison figure for paper.
Creates Figure 1: Survival rates across SOFA severity strata.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from SOFA-stratified evaluation (verified from pkl files)
data = {
    'Random': [98.6, 100.0, 87.5],
    'Heuristic': [100.0, 98.4, 88.1],
    'BC': [100.0, 96.7, 88.9],
    'CQL': [100.0, 100.0, 84.6]
}

categories = ['Low SOFA\n(<-0.45)', 'Medium SOFA\n(-0.45 to 0.21)', 'High SOFA\n(>0.21)']
x = np.arange(len(categories))
width = 0.2

# Create figure with academic styling
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-paper')

colors = ['#d3d3d3', '#4a90e2', '#50c878', '#ff6b6b']
for i, (policy, values) in enumerate(data.items()):
    plt.bar(x + i*width - 1.5*width, values, width, label=policy,
            color=colors[i], edgecolor='black', linewidth=1)

plt.xlabel('SOFA Severity Category', fontsize=14, fontweight='bold')
plt.ylabel('Survival Rate (%)', fontsize=14, fontweight='bold')
plt.title('Survival Rates Across SOFA Severity Strata', fontsize=16, fontweight='bold')
plt.xticks(x, categories, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(50, 102)
plt.legend(fontsize=11, loc='lower left', frameon=True, edgecolor='black')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal line at 100% for reference
plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add annotations for key findings
plt.annotate('CQL & Random:\n100%',
            xy=(1, 100), xytext=(1.3, 105),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.annotate('BC: 88.9%\n(Best on High SOFA)',
            xy=(2, 88.9), xytext=(1.5, 75),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#50c878', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'stratified_survival_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print('[SUCCESS] Figure saved: stratified_survival_comparison.png')

# plt.show()  # Comment out to avoid blocking in script execution
