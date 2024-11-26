import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from rdkit import Chem
import os
import glob
import pandas as pd
from collections import defaultdict
from preprocessing.dock import get_protein_mol_att
from sklearn.preprocessing import MinMaxScaler
from vis.utils import get_residue_ids_from_pdb, replace_local_pathes

v_cmap = plt.get_cmap("viridis")
TAB10_COLORS = plt.get_cmap("tab10").colors


def create_pymol_script_with_sdf(pdb_file: str, sdf_files: list, color_values,
                                 output_script: str = "protein_molecules_colored.pml"):
    residue_ids = get_residue_ids_from_pdb(pdb_file)
    with open(output_script, 'w') as f:
        f.write(f"load {pdb_file}, protein\n")
        r, g, b = TAB10_COLORS[0]
        for i, sdf_file in enumerate(sdf_files):
            molecule_name = f"molecule_{i + 1}"
            f.write(f"load {sdf_file}, {molecule_name}\n")
            f.write(f"show sticks, {molecule_name}\n")
            f.write(f"color red,{molecule_name} \n")
        for (chain_id, res_num), value in zip(residue_ids, color_values):
            r, g, b, _ = v_cmap(value)
            f.write(f"set_color color_{chain_id}_{res_num}, [{r}, {g}, {b}]\n")
            f.write(f"color color_{chain_id}_{res_num}, protein and chain {chain_id} and resi {res_num}\n")
        f.write("show cartoon, protein\n")
    print(f"PyMOL script '{output_script}' created successfully.")


def remove_stereo(s):
    s = s.replace(" ", "")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    print(Chem.MolToSmiles(mol))
    return Chem.MolToSmiles(mol)


def load_maps():
    with open("datasets/docking/smiles_to_id.txt") as f:
        id_to_smile = {int(x.split()[1]): x.split()[0] for x in f.readlines()}
        smile_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
    ec_mapping = pd.read_csv("datasets/ec_map.csv")
    uniport_to_ec = defaultdict(str)
    for i, row in ec_mapping.iterrows():
        uniport_to_ec[row["Uniprot_id"]] = row["EC_full"]
    return id_to_smile, smile_to_id, uniport_to_ec


def remove_dup_mis_mols(molecules_ids):
    molecules_smiles = [id_to_smile[int(x)] for x in molecules_ids]
    molecules_smiles_no_stereo = [remove_stereo(x) for x in molecules_smiles]
    molecules_smiles_mask = [True] * len(molecules_smiles)
    for i in range(1, len(molecules_smiles)):
        if molecules_smiles_no_stereo[i] is None:
            molecules_smiles_mask[i] = False
        if molecules_smiles_no_stereo[i] in molecules_smiles_no_stereo[:i]:
            molecules_smiles_mask[i] = False
    return [molecules_ids[i] for i in range(len(molecules_ids)) if molecules_smiles_mask[i]]


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--protein_id", type=str, default="A0A009H5L7")
args = parser.parse_args()
protein_id = args.protein_id

id_to_smile, smile_to_id, uniport_to_ec = load_maps()

protein_ec = uniport_to_ec[protein_id]
pdb_file = f"datasets/pdb_files/{protein_id}/{protein_id}_esmfold.pdb"
molecules_ids = os.listdir(f"datasets/docking2/{protein_id}")
print(f"Found {len(molecules_ids)} molecules for protein {protein_id}")
molecules_ids = remove_dup_mis_mols(molecules_ids)
sdf_files = []
for m in molecules_ids:
    sdf_files.extend(glob.glob(f"datasets/docking2/{protein_id}/{m}/complex_0/*.sdf"))
    print(f"Found {len(molecules_ids)} unique molecules for protein {protein_id}")
    docking_attention_emd, w = get_protein_mol_att(protein_id, m, 0.9, True, return_weights=True)
    w = MinMaxScaler(feature_range=(0, 1)).fit_transform(w.reshape(-1, 1)).flatten()
    output_script = f"vis/scripts/protein_molecules_colored_{m}.pml"
    create_pymol_script_with_sdf(pdb_file, sdf_files, w, output_script=output_script)
    replace_local_pathes(output_script)
