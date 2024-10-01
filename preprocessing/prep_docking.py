import pandas as pd
from collections import defaultdict
from rdkit import Chem
import os
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")
ec_mapping = pd.read_csv("datasets/ec_map.csv")
ec_to_uniport = defaultdict(str)
for i, row in ec_mapping.iterrows():
    ec_to_uniport[row["EC_full"]] = row["Uniprot_id"]

base_dataset = "datasets/ecreact/ecreact-1.0.txt"
with open(base_dataset) as f:
    lines = f.readlines()
smiles_to_id = dict()
all_names = set()
results = []
for line in lines:
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
        name = f"{uniprot_id}_{smiles_to_id[s]}"
        if name in all_names:
            continue
        all_names.add(name)
        results.append((name, s, f'../BioCatalystChem/datasets/pdb_files/{uniprot_id}/{uniprot_id}_esmfold.pdb'))
results_df = pd.DataFrame(results, columns=["complex_name", "ligand_description", "protein_path"])
output_pdb_dir = "datasets/docking"
if not os.path.exists(output_pdb_dir):
    os.makedirs(output_pdb_dir)
n_splits = 8
for i in range(n_splits):
    results_df.iloc[i::n_splits].to_csv(f"{output_pdb_dir}/split_{i + 1}.csv", index=False)
with open(f"{output_pdb_dir}/smiles_to_id.txt") as f:
    for k, v in smiles_to_id.items():
        f.write(f"{k} {v}\n")