import os
from collections import Counter

base_dataset = "../datasets/ecreact/level4"
output_dir = "../datasets/ecreact/level4_duplicates"
os.makedirs(output_dir, exist_ok=True)
srcs_counter = Counter()
for split in ["train", "valid", "test"]:
    with open(f"{base_dataset}/src-{split}.txt") as f:
        src_lines = f.read().splitlines()
        src = [x.split("|")[0] for x in src_lines]
        srcs_counter.update(src)
print(len(srcs_counter))


for split in ["train", "valid", "test"]:
    with open(f"{base_dataset}/src-{split}.txt") as f:
        src_lines = f.read().splitlines()
    with open(f"{base_dataset}/tgt-{split}.txt") as f:
        tgt_lines = f.read().splitlines()
    filter_src = []
    filter_tgt = []
    assert len(src_lines) == len(tgt_lines)
    for src, tgt in zip(src_lines, tgt_lines):
        src_id = src.split("|")[0]
        if srcs_counter[src_id] > 1:
            filter_src.append(src)
            filter_tgt.append(tgt)
    assert len(filter_src) == len(filter_tgt)
    with open(f"{output_dir}/src-{split}.txt", "w") as f:
        f.write("\n".join(filter_src))
    with open(f"{output_dir}/tgt-{split}.txt", "w") as f:
        f.write("\n".join(filter_tgt))
    print(f"Split: {split} - {len(filter_src)} / {len(src_lines)}")
