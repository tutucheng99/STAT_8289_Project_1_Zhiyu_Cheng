import json

# Load notebook
with open(r'C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\notebooks\LEG_interplate_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract evaluator class code
print("=" * 80)
print("SEARCHING FOR EVALUATOR CLASS AND EVALUATION LOGIC")
print("=" * 80)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Look for evaluator class or evaluation functions
        if any(keyword in source.lower() for keyword in ['class evaluator', 'def evaluate', 'survival_rate', 'env.reset', 'env.step']):
            # Skip if it's mostly image data
            if 'iVBORw0KGgo' not in source and len(source) < 10000:
                print(f"\n{'='*80}")
                print(f"CELL {i+1}:")
                print(f"{'='*80}")
                # Print first 2000 chars to avoid overwhelming output
                print(source[:2000])
                if len(source) > 2000:
                    print("\n... [truncated] ...")
                print()
