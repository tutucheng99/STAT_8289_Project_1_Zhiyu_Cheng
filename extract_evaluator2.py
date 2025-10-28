import json

# Load notebook
with open(r'C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\notebooks\LEG_interplate_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}\n")

# Print each cell's first 500 chars to understand structure
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Skip image data
        if 'iVBORw0KGgo' not in source:
            print(f"\n{'='*80}")
            print(f"CELL {i+1} (Code, {len(source)} chars):")
            print(f"{'='*80}")
            print(source[:800])
            if len(source) > 800:
                print("\n... [content truncated] ...")
    elif cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if len(source) < 500:
            print(f"\n{'='*80}")
            print(f"CELL {i+1} (Markdown):")
            print(f"{'='*80}")
            print(source)
