from collections import Counter
import os
import pandas as pd

# Read test files
first_test_src_file = "datasets/ecreact/level4/src-test.txt"
first_test_tgt_file = "datasets/ecreact/level4/tgt-test.txt"
src_tgt_to_ec = dict()

with open(first_test_src_file) as f:
    first_test_src = f.read().splitlines()
with open(first_test_tgt_file) as f:
    first_test_tgt = f.read().splitlines()

for src, tgt in zip(first_test_src, first_test_tgt):
    ec = src.split(" | ")[1]
    src = src.split(" | ")[0]
    src_tgt_to_ec[src + " >> " + tgt] = ec

with open("datasets/ecreact/level4/src-train.txt") as f:
    train_src = f.read().splitlines()
with open("datasets/ecreact/level4/tgt-train.txt") as f:
    train_tgt = f.read().splitlines()

# Process results
files = os.listdir("results/full")
full_res = pd.DataFrame(index=list(src_tgt_to_ec.keys()), columns=files)
for file_name in files:
    if not file_name.endswith(".csv"):
        continue
    with open("results/full/" + file_name) as f:
        lines = f.read().splitlines()
        for line in lines:
            src, tgt, res = line.split(",")
            src = src.split(" | ")[0]
            id_ = src + " >> " + tgt
            full_res.loc[id_, file_name] = res == "True"

print(full_res.shape)
full_res = full_res.dropna()
print(full_res.shape)
print(full_res.mean(axis=0).sort_values().to_frame())
