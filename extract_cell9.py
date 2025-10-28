import json
import sys

# Load notebook
with open(r'C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\notebooks\LEG_interplate_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract Cell 9 (index 8)
cell = nb['cells'][8]
code = ''.join(cell['source'])

# Write to file to avoid encoding issues
with open('cell9_code.txt', 'w', encoding='utf-8') as f:
    f.write(code)

print("Cell 9 code extracted to cell9_code.txt")
print(f"Total length: {len(code)} characters")
