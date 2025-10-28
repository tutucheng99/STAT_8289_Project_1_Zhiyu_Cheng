import pickle
import numpy as np
import os
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Load the data
data_path = 'data/offline_dataset.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")

# Check structure
if isinstance(data, dict):
    print(f"Data keys: {list(data.keys())}")
    if 'actions' in data:
        all_actions = data['actions']
    elif 'observations' in data:
        all_actions = data['actions']
    else:
        print("Checking first item structure...")
        first_key = list(data.keys())[0]
        print(f"First key: {first_key}, type: {type(data[first_key])}")
        all_actions = None
elif isinstance(data, list):
    print(f"Data is list with {len(data)} items")
    if len(data) > 0:
        print(f"First item type: {type(data[0])}")
        if isinstance(data[0], dict):
            print(f"First item keys: {list(data[0].keys())}")
            actions = [ep['actions'] for ep in data]
            all_actions = np.concatenate(actions)
        else:
            all_actions = None
else:
    all_actions = None

if all_actions is not None:
    print('\nUnique actions in data:', sorted(np.unique(all_actions).astype(int)))
    print(f'\nTotal actions: {len(all_actions)}')
    print(f'Number of unique actions: {len(np.unique(all_actions))}')

    print('\nAction counts:')
    unique, counts = np.unique(all_actions, return_counts=True)
    for a, c in sorted(zip(unique, counts)):
        print(f'  Action {int(a):2d}: {c:5d} ({c/len(all_actions)*100:.2f}%)')

    # Check for action 24
    if 24 in unique:
        print('\n*** Action 24 found in data! ***')
    else:
        print('\n*** Action 24 NOT found in data (confirms 24-action space) ***')

    # Decode actions to (IV, VP) pairs
    print('\n\nDecoding actions to (IV, VP) pairs:')
    for action in sorted(np.unique(all_actions).astype(int)):
        iv = action // 5
        vp = action % 5
        print(f'  Action {action:2d} = (IV={iv}, VP={vp})')
else:
    print("Could not extract actions from data")
