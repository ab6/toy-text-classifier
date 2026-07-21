import json
import os
import pandas as pd
from datasets import load_dataset

# 1. Define dummy data
data_train = [
    {"text": "I absolutely love this library!", "label": 1},
    {"text": "This is completely broken.", "label": 0},
    {"text": "Standard performance, nothing special.", "label": 1},
]

data_test = [
    {"text": "Highly recommended for developers.", "label": 1},
    {"text": "Worst experience ever.", "label": 0},
]

# 2. Create a temporary folder for mock data
os.makedirs("dummy_data", exist_ok=True)

# --- Option A: Testing with CSV files ---
pd.DataFrame(data_train).to_csv("dummy_data/train.csv", index=False)
pd.DataFrame(data_test).to_csv("dummy_data/test.csv", index=False)

# --- Option B: Testing with JSON Lines (JSONL) files ---
with open("dummy_data/train.jsonl", "w") as f:
    for item in data_train:
        f.write(json.dumps(item) + "\n")

with open("dummy_data/test.jsonl", "w") as f:
    for item in data_test:
        f.write(json.dumps(item) + "\n")

print("Dummy CSV and JSONL files generated successfully inside 'dummy_data/'!")
