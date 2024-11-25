from tqdm import tqdm
from preprocessing.build_tokenizer import redo_ec_split
from preprocessing.ec_to_vec import EC2Vec
# from preprocessing.dock import get_reaction_attention_emd
from preprocessing.dock import Docker
from collections import defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

n_cpu = os.cpu_count()

V2 = True

ec_to_vec = EC2Vec(load_model=False)
with open("datasets/docking/smiles_to_id.txt") as f:
    smiles_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
ec_mapping = pd.read_csv("datasets/ec_map.csv")
ec_to_uniprot = defaultdict(str)
for i, row in ec_mapping.iterrows():
    ec_to_uniprot[row["EC_full"]] = row["Uniprot_id"]
input_base = "datasets/ecreact/level4"

doker = Docker(0.5, V2)


def process_reaction(text, ec):
    return doker.dock_src_ec(text, ec)
    # return get_reaction_attention_emd(text, ec, ec_to_uniprot, smiles_to_id, alpha=0.5, v2=True)


for split in ["test", "train", "valid"]:
    with open(f"{input_base}/src-{split}.txt") as f:
        first_src_lines = f.read().splitlines()

    with open(f"{input_base}/tgt-{split}.txt") as f:
        first_tgt_lines = f.read().splitlines()

    assert len(first_src_lines) == len(first_tgt_lines)

    src_ec = [redo_ec_split(text, True) for text in first_src_lines]
    src_lines = [x[0] for x in src_ec]
    ec_lines = [x[1] for x in src_ec]
    pre_lines = [ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]

    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        dae_lines = list(tqdm(executor.map(process_reaction, src_lines, ec_lines), total=len(src_lines)))

    # dae_lines = [
    #     get_reaction_attention_emd(text, ec, ec_to_uniprot, smiles_to_id, alpha=0.5)
    #     for text, ec in tqdm(zip(src_lines, ec_lines), total=len(src_lines))
    # ]
    final_src_lines = []
    final_tgt_lines = []
    removed_lines = 0
    for i in range(len(first_src_lines)):
        if (pre_lines[i] is not None) and (dae_lines[i] is not None):
            final_src_lines.append(first_src_lines[i])
            final_tgt_lines.append(first_tgt_lines[i])
        else:
            removed_lines += 1
    print(
        f"Removed {removed_lines} samples (out of {len(first_src_lines)})[{removed_lines / len(first_src_lines) * 100}%] from {split}")

    with open(f"{input_base}/src-{split}.txt", "w") as f:
        for line in final_src_lines:
            f.write(line + "\n")
    with open(f"{input_base}/tgt-{split}.txt", "w") as f:
        for line in final_tgt_lines:
            f.write(line + "\n")
