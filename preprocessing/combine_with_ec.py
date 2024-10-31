from os.path import join as pjoin
import os
from typing import Dict, Tuple


def dataset_to_dict(dataset: str, split: str) -> Dict[Tuple[str, str], str]:
    src_file = pjoin(dataset, f"src-{split}.txt")
    tgt_file = pjoin(dataset, f"tgt-{split}.txt")
    with open(src_file) as f:
        src_lines = f.read().splitlines()
    with open(tgt_file) as f:
        tgt_lines = f.read().splitlines()
    assert len(src_lines) == len(tgt_lines)
    mapping = {}
    for src, tgt in zip(src_lines, tgt_lines):
        src, encode = src.split(" | ")
        key = (src, tgt)
        mapping[key] = encode
    return mapping


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    output_dict = {}
    for key in dict1:
        if key in dict2:
            output_dict[key] = (dict2, dict1[key])
    return output_dict


def write_data_dict(data_dict, output_base, split):
    os.makedirs(output_base, exist_ok=True)
    src_file = pjoin(output_base, f"src-{split}.txt")
    tgt_file = pjoin(output_base, f"tgt-{split}.txt")
    src_list = []
    tgt_list = []
    for key in data_dict:
        src, tgt = key
        ec, encode = data_dict[key]
        src_list.append(f"{src} | {ec} | {encode}")
        tgt_list.append(tgt)
    with open(src_file, "w") as f:
        f.write("\n".join(src_list))
    with open(tgt_file, "w") as f:
        f.write("\n".join(tgt_list))


ec_dataset = "datasets/ecreact/level4"
target_dataset = "datasets/ecreact/quant_2-0.5_0_5_5_5"
output_dataset = target_dataset + "_plus"
os.makedirs(output_dataset, exist_ok=True)
for split in ["train", "test", "valid"]:
    ec_dict = dataset_to_dict(ec_dataset, split)
    target_dict = dataset_to_dict(target_dataset, split)
    output_dict = merge_dicts(ec_dict, target_dict)
    write_data_dict(output_dict, output_dataset, split)
