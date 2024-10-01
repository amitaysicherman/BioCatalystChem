import pandas as pd
from collections import defaultdict
from rdkit import Chem
import os

ec_mapping = pd.read_csv("datasets/ec_map.csv")
ec_to_uniport = defaultdict(str)
for i, row in ec_mapping.iterrows():
    ec_to_uniport[row["EC_full"]] = row["Uniprot_id"]

base_dataset = "datasets/ecreact/ecreact-1.0.txt"
with open(base_dataset) as f:
    lines = f.readlines()

results = []
for line in lines:
    ec = line.split("|")[1].split(">>")[0]
    uniprot_id = ec_to_uniport[ec]
    input_smiles = line.split("|")[0]
    input_smiles = [x for x in input_smiles.split(".") if len(x) > 3]
    for s in input_smiles:
        # remove stereochemistry
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        s = Chem.MolToSmiles(mol)
        results.append((uniprot_id, s, f'../BioCatalystChem/datasets/pdb_files/{uniprot_id}/{uniprot_id}_esmfold.pdb'))
results_df = pd.DataFrame(results, columns=["complex_name", "ligand_description", "protein_path"])
output_pdb_dir = "datasets/docking"
if not os.path.exists(output_pdb_dir):
    os.makedirs(output_pdb_dir)
n_splits = 8
for i in range(n_splits):
    results_df.iloc[i::n_splits].to_csv(f"{output_pdb_dir}/split_{i + 1}.csv", index=False)
