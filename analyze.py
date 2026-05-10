import pandas as pd
from pathlib import Path

DATA = Path("data/runs.jsonl")

if not DATA.exists():
    print("No data found at data/runs.jsonl")
    exit()

df = pd.read_json(DATA, lines=True)

print("\n=== TOTAL RUNS ===")
print(len(df))

print("\n=== BY FRICTION ===")
print(df.groupby("friction")["outcome"].value_counts())

print("\n=== BY GRID SIZE ===")
print(df.groupby("grid_size")["outcome"].value_counts())

print("\n=== AVERAGE BLOOM SCORE BY FRICTION ===")
print(df.groupby("friction")["bloom_score"].mean().sort_values(ascending=False))
