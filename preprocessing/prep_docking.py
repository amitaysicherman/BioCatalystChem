import pandas as pd
from collections import defaultdict
from rdkit import Chem
import os
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from tqdm import tqdm

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")
ec_mapping = pd.read_csv("datasets/ec_map.csv")
ec_to_uniport = defaultdict(str)
ec_to_fasta = defaultdict(str)
for i, row in ec_mapping.iterrows():
    ec_to_uniport[row["EC_full"]] = row["Uniprot_id"]
    ec_to_fasta[row["EC_full"]] = row["Sequence"]

base_dataset = "datasets/ecreact/ecreact-1.0.txt"
with open(base_dataset) as f:
    lines = f.readlines()
smiles_to_id = dict()
all_names = set()
results = []
for line in tqdm(lines):
    ec = line.split("|")[1].split(">>")[0]
    uniprot_id = ec_to_uniport[ec]
    input_smiles = line.split("|")[0]
    input_smiles = [x for x in input_smiles.split(".") if len(x) > 3]
    for s in input_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        s = Chem.MolToSmiles(mol)
        if s not in smiles_to_id:
            smiles_to_id[s] = len(smiles_to_id)
        name = f"{uniprot_id}/{smiles_to_id[s]}"
        if name in all_names:
            continue
        all_names.add(name)
        pdb_file = f"../BioCatalystChem/datasets/pdb_files/{uniprot_id}/{uniprot_id}_esmfold.pdb"
        if not os.path.exists(pdb_file):
            continue
        results.append((name, s, pdb_file, ""))
results_df = pd.DataFrame(results, columns=["complex_name", "ligand_description", "protein_path", "protein_sequence"])
output_pdb_dir = "../BioCatalystChem/datasets/docking/"
if not os.path.exists(output_pdb_dir):
    os.makedirs(output_pdb_dir)
with open(f"{output_pdb_dir}/smiles_to_id.txt", 'w') as f:
    for k, v in smiles_to_id.items():
        f.write(f"{k} {v}\n")

cmds = []
base_cmd = "python -m inference --config default_inference_args.yaml"
skip_1 = 0
skip_2 = 0
pbar = tqdm(results_df.iterrows(), total=len(results_df))
for i, row in pbar:
    name = row["complex_name"]
    pdb_file = row["protein_path"]
    ligand = row["ligand_description"]
    output_dir = output_pdb_dir + name
    final_output = output_dir + "/complex_0"
    if os.path.exists(final_output):
        files_in_dir = os.listdir(final_output)
        if any(f.startswith("rank10") for f in files_in_dir):
            skip_1 += 1
            continue
        if not files_in_dir or not any(f.startswith("rank1") for f in files_in_dir):
            skip_2 += 1
            continue
    else:
        skip_2 += 1

    cmds.append(
        f"{base_cmd} --protein_path '{pdb_file}' --ligand '{ligand}' --out_dir '{output_dir}'")
    pbar.set_description(f"s1: {skip_1}, s2: {skip_2}, r: {len(cmds)}")
print(f"Skipped {skip_1} docking runs")
print(f"Skipped {skip_2} docking runs")
print(f"Running {len(cmds)} docking runs")
scripts_dir = "../DiffDock/scripts"
os.makedirs(scripts_dir, exist_ok=True)

n_splits = 200
for i in range(n_splits):
    with open(f"{scripts_dir}/run_{i + 1}.sh", 'w') as f:
        for cmd in cmds[i::n_splits]:
            f.write(f"{cmd}\n")
