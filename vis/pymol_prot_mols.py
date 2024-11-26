import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from preprocessing.dock import get_protein_mol_att
from sklearn.preprocessing import MinMaxScaler
from vis.utils import get_residue_ids_from_pdb, replace_local_pathes, load_maps, remove_dup_mis_mols, \
    filter_molecule_by_len
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
v_cmap = plt.get_cmap("Greens")
TAB10_COLORS = plt.get_cmap("tab10").colors


def create_pymol_script_with_sdf(pdb_file: str, sdf_files: list, color_values,
                                 output_script: str = "protein_molecules_colored.pml"):
    residue_ids = get_residue_ids_from_pdb(pdb_file)
    with open(output_script, 'w') as f:
        f.write(f"load {pdb_file}, protein\n")
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--protein_id", type=str, default="A0A009HWM5")
args = parser.parse_args()
protein_id = args.protein_id

id_to_smile, smile_to_id, uniport_to_ec = load_maps()

protein_ec = uniport_to_ec[protein_id]
pdb_file = f"datasets/pdb_files/{protein_id}/{protein_id}_esmfold.pdb"
molecules_ids = os.listdir(f"datasets/docking2/{protein_id}")
c1 = len(molecules_ids)
molecules_ids = remove_dup_mis_mols(molecules_ids, id_to_smile)
c2 = len(molecules_ids)
print(f"Found {c1} molecules for protein {protein_id}, after removing duplicates: {c2}")
for m in molecules_ids:
    sdf_files = glob.glob(f"datasets/docking2/{protein_id}/{m}/complex_0/*.sdf")
    mc1=len(sdf_files)
    sdf_files = filter_molecule_by_len(sdf_files, 0.5)
    mc2=len(sdf_files)
    print(f"Found {mc1} molecules for protein {protein_id}, after filtering by length: {mc2}")
    docking_attention_emd, w = get_protein_mol_att(protein_id, m, 0.9, True, return_weights=True)

    plt.figure(figsize=(10, 2))
    plt.plot(w)
    # remove grid and axis
    plt.grid(False)
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(f"vis/figures/protein_molecules_{protein_id}_{m}.png", dpi=300, bbox_inches='tight')
    w = np.log(w)

    w = MinMaxScaler(feature_range=(0, 1)).fit_transform(w.reshape(-1, 1)).flatten()
    output_script = f"vis/scripts/protein_molecules_{protein_id}_{m}.pml"

    create_pymol_script_with_sdf(pdb_file, sdf_files, w, output_script=output_script)
    replace_local_pathes(output_script)
