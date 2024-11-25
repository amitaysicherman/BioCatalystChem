import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from preprocessing.dock import get_reaction_attention_emd
from preprocessing.build_tokenizer import redo_ec_split


def args_to_file(v2, alpha):
    docking_dir = "docking2" if v2 else "docking"
    return "datasets/" + docking_dir + f"/docking_{alpha}.txt"


def load_docking_file(v2, alpha):
    docking_dir = "docking2" if v2 else "docking"
    with open("datasets/" + docking_dir + f"/docking_{alpha}.txt") as f:
        lines = f.read().splitlines()
    src_ec_to_vec = dict()
    for line in lines:
        src, ec, v_str = line.split(",")
        key = (src, ec)
        v = [float(x) for x in v_str.split()]
        src_ec_to_vec[key] = np.array(v)
    return src_ec_to_vec


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=int, default=50)
    parser.add_argument("--v2", type=int, default=1)
    args = parser.parse_args()
    args.alpha = float(args.alpha / 100)

    with open("datasets/docking/smiles_to_id.txt") as f:
        smiles_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}

    ec_mapping = pd.read_csv("datasets/ec_map.csv")
    ec_to_uniprot = defaultdict(str)
    for i, row in ec_mapping.iterrows():
        ec_to_uniprot[row["EC_full"]] = row["Uniprot_id"]

    with open("datasets/ecreact/level4/src-train.txt") as f:
        src_lines = f.read().splitlines()

    with open("datasets/ecreact/level4/src-valid.txt") as f:
        src_lines += f.read().splitlines()

    with open("datasets/ecreact/level4/src-test.txt") as f:
        src_lines += f.read().splitlines()

    src_ec_to_vec = dict()
    for text in tqdm(src_lines):
        src, ec = redo_ec_split(text, True)

        key = (src, ec)
        if key in src_ec_to_vec:
            continue
        v = get_reaction_attention_emd(src, ec, ec_to_uniprot, smiles_to_id, alpha=args.alpha, v2=args.v2)
        if v is not None:
            src_ec_to_vec[key] = v
    with open(args_to_file(args.v2, args.alpha), "w") as f:
        for key, v_numpy in src_ec_to_vec.items():
            v_str = " ".join(str(x) for x in v_numpy)
            f.write(f"{key[0]},{key[1]},{v_str}\n")
