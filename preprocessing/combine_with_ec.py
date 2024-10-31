from os.path import join as pjoin
import os
from typing import Dict, Tuple
from tqdm import tqdm


def dataset_to_dict(dataset: str, split: str) -> Dict[Tuple[str, str], str]:
    src_file = pjoin(dataset, f"src-{split}.txt")
    tgt_file = pjoin(dataset, f"tgt-{split}.txt")
    with open(src_file) as f:
        src_lines = f.read().splitlines()
    with open(tgt_file) as f:
        tgt_lines = f.read().splitlines()
    assert len(src_lines) == len(tgt_lines)
    mapping = {}
    for src, tgt in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
        src, encode = src.split(" | ")
        key = (src, tgt)
        mapping[key] = encode
    return mapping


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    output_dict = {}
    match_count = 0
    for key in dict1:
        if key in dict2:
            output_dict[key] = (dict2[key], dict1[key])
            match_count += 1
    print(f"Matched {match_count} out of {len(dict1)} ({match_count / len(dict1):.1%})")
    return output_dict


def write_data_dict(data_dict, output_base, split):
    os.makedirs(output_base, exist_ok=True)
    src_file = pjoin(output_base, f"src-{split}.txt")
    tgt_file = pjoin(output_base, f"tgt-{split}.txt")

    src_tgt_list = [[(src, ec, encode), tgt] for (src, tgt), (ec, encode) in data_dict.items()]
    print(f"Writing {len(src_tgt_list)} samples to {src_file} and {tgt_file}")
    with open(src_file, "w") as f:
        with open(tgt_file, "w") as g:
            for s, tgt in src_tgt_list:
                f.write(" | ".join(s) + "\n")
                g.write(tgt + "\n")


ec_dataset = "datasets/ecreact/level4"
target_dataset = "datasets/ecreact/quant_2_5_6_10"
output_dataset = target_dataset + "_plus"
os.makedirs(output_dataset, exist_ok=True)
for split in ["train", "test", "valid"]:
    ec_dict = dataset_to_dict(ec_dataset, split)
    target_dict = dataset_to_dict(target_dataset, split)
    output_dict = merge_dicts(target_dict, ec_dict)
    write_data_dict(output_dict, output_dataset, split)
