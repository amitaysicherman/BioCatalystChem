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
sns.set_style("white")
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


def plot_w(w, protein_id, m):
    fig = plt.figure(figsize=(10, 2))
    plt.plot(w)
    # remove grid and axis
    plt.grid(False)
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(f"vis/figures/protein_molecules_{protein_id}_{m}.png", bbox_inches='tight')
    plt.close(fig)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--protein_id", type=str, default=["B5UAT8", "F0E1K6"], nargs="+")
args = parser.parse_args()
id_to_smile, smile_to_id, uniport_to_ec = load_maps()

all_vecs = []

for protein_id in args.protein_id:
    protein_vecs = []
    protein_ec = uniport_to_ec[protein_id]
    pdb_file = f"datasets/pdb_files/{protein_id}/{protein_id}_esmfold.pdb"
    molecules_ids = os.listdir(f"datasets/docking2/{protein_id}")
    c1 = len(molecules_ids)
    molecules_ids = remove_dup_mis_mols(molecules_ids, id_to_smile)
    c2 = len(molecules_ids)
    print(f"Found {c1} molecules for protein {protein_id}, after removing duplicates: {c2}")
    for m in molecules_ids:
        sdf_files = glob.glob(f"datasets/docking2/{protein_id}/{m}/complex_0/*.sdf")
        mc1 = len(sdf_files)
        sdf_files = filter_molecule_by_len(sdf_files, 0.5)
        mc2 = len(sdf_files)
        print(f"Found {mc1} molecules for protein {protein_id}, after filtering by length: {mc2}")
        docking_attention_emd, w = get_protein_mol_att(protein_id, m, 0.9, True, return_weights=True)
        if len(protein_vecs) == 0:
            protein_vecs.append(get_protein_mol_att(protein_id, m, 0.0, True, return_weights=False))
        protein_vecs.append(docking_attention_emd)
        plot_w(w, protein_id, m)
        w = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.log(w).reshape(-1, 1)).flatten()
        output_script = f"vis/scripts/protein_molecules_{protein_id}_{m}.pml"
        create_pymol_script_with_sdf(pdb_file, sdf_files, w, output_script=output_script)
        replace_local_pathes(output_script)
    all_vecs.append(protein_vecs)

# create 2D plot of protein embeddings with TSNE
from sklearn.manifold import TSNE

per_protein_vecs = [len(x) for x in all_vecs]
vecs_concat = np.array(sum(all_vecs, []))
print(vecs_concat.shape)
protein_names = [f'{p}-({uniport_to_ec[p]}' for p in args.protein_id]
vecs_2d = TSNE(n_components=2).fit_transform(vecs_concat)

fig = plt.figure(figsize=(7, 7))
for i, name in enumerate(protein_names):
    plt.scatter(vecs_2d[0][sum(per_protein_vecs[:i]):sum(per_protein_vecs[:i + 1]), 0],
                vecs_2d[0][sum(per_protein_vecs[:i]):sum(per_protein_vecs[:i + 1]), 1],
                label=name, color=TAB10_COLORS[i])
plt.legend()
plt.title("Protein Molecules Embeddings")
plt.tight_layout()
plt.axis('off')
plt.grid(False)
names = " + ".join(protein_names)
plt.savefig(f"vis/figures/protein_molecules_tsne_{names}.png", bbox_inches='tight')

plt.tight_layout()
plt.savefig(f"vis/figures/protein_molecules_{protein_id}_tsne.png", bbox_inches='tight')
plt.close(fig)
