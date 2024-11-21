from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from collections import defaultdict
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import os

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")
ec_mapping = pd.read_csv("datasets/ec_map.csv")
ec_to_uniport = defaultdict(str)
ec_to_fasta = defaultdict(str)
for i, row in ec_mapping.iterrows():
    ec_to_uniport[row["EC_full"]] = row["Uniprot_id"]
    ec_to_fasta[row["EC_full"]] = row["Sequence"]



def line_to_mol_id(line: str, smiles_to_id: dict):
    input_smiles = line.split("|")[0]
    input_smiles = [x for x in input_smiles.split(".") if len(x) > 3]
    ids = []
    smiles = []
    for s in input_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        s = Chem.MolToSmiles(mol)
        if s not in smiles_to_id:
            smiles_to_id[s] = len(smiles_to_id)
        ids.append(smiles_to_id[s])
        smiles.append(s)
    return ids, smiles


if __name__ == "__main__":
    skip_count = 0
    skip_count2 = 0
    with open("datasets/docking/smiles_to_id.txt") as f:
        lines= f.readlines()
    smiles_to_id = dict()

    for line in lines:
        s, i = line.strip().split(" ")
        smiles_to_id[s] = int(i)

    all_names = set()
    base_dataset = "datasets/ecreact/ecreact-1.0.txt"
    with open(base_dataset) as f:
        lines = f.readlines()
    results = []
    pbar = tqdm(lines, total=len(lines))
    for line in pbar:
        ec = line.split("|")[1].split(">>")[0]
        uniprot_id = ec_to_uniport[ec]
        pdb_file = f"../BioCatalystChem/datasets/pdb_files/{uniprot_id}/{uniprot_id}_esmfold.pdb"
        ids, smiles = line_to_mol_id(line, smiles_to_id)
        for i, s in zip(ids, smiles):
            name = f"../BioCatalystChem/datasets/docking2/{uniprot_id}/{i}/complex_0"
            if os.path.exists(name) and len(os.listdir(name)) > 0:
                skip_count2 += 1
            prev_run_name = f"datasets/docking/{uniprot_id}/{i}/complex_0"
            if not os.path.exists(prev_run_name) or len(os.listdir(prev_run_name)) == 0:
                skip_count += 1
                continue
            if name in all_names:
                continue
            all_names.add(name)
            results.append([name, pdb_file, s, ""])
        pbar.set_description(f"Skipped {skip_count} entries / {skip_count2} entries, {len(results)} entries found")
    df = pd.DataFrame(results, columns=["complex_name", "protein_path", "ligand_description", "protein_sequence"])
    df = df.sort_values(by="complex_name")

    n_splits = 15
    split_size = len(df) // n_splits
    output_base = "../DiffDock/input_csvs"
    os.makedirs(output_base, exist_ok=True)
    for i in range(n_splits):
        df.iloc[i * split_size:(i + 1) * split_size].to_csv(f"{output_base}/protein_ligand_{i}.csv", index=False)

    with open(f"{output_base}/smiles_to_id_2.txt", 'w') as f:
        for k, v in smiles_to_id.items():
            f.write(f"{k} {v}\n")
