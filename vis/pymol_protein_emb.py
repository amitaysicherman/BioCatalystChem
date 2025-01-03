import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from vis.utils import get_residue_ids_from_pdb, replace_local_pathes


def create_pymol_script(pdb_file: str, embedding_file: str, output_script, k_clusters=None):
    embeddings = np.load(embedding_file)[1:-1]  # Assuming shape is (num_residues, embedding_dimension)
    residue_ids = get_residue_ids_from_pdb(pdb_file)
    assert len(residue_ids) == embeddings.shape[0], "Mismatch between PDB residues and embeddings"

    if k_clusters:
        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_
        cluster_colors_map = plt.cm.get_cmap("tab10")
        emb_colors = cluster_colors_map(cluster_labels)
        emb_colors = emb_colors[:, :3]

    else:

        pca = PCA(n_components=1)
        pca_values = pca.fit_transform(embeddings).flatten()  # Flatten to get a 1D array
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_pca_values = scaler.fit_transform(pca_values.reshape(-1, 1)).flatten()
        viridis_colors = plt.cm.get_cmap("viridis")
        emb_colors = viridis_colors(normalized_pca_values)
        emb_colors = emb_colors[:, :3]

    with open(output_script, 'w') as f:
        # Load the PDB file in PyMOL
        f.write(f"load {pdb_file}, protein\n")
        for (chain_id, res_num), color in zip(residue_ids, emb_colors):
            r, g, b = color  # RGB values
            f.write(f"set_color color_{chain_id}_{res_num}, [{r}, {g}, {b}]\n")
            f.write(f"color color_{chain_id}_{res_num}, protein and chain {chain_id} and resi {res_num}\n")
        f.write('show_as("mesh"      ,"all")')
    print(f"PyMOL script '{output_script}' created successfully.")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--protein_id", type=str, default=["E9P8D2", "F0E1K6","H2K888"], nargs="+")
args = parser.parse_args()
k = args.k
for protein_id in args.protein_id:
    output_script = f"vis/scripts/protein_emb_{protein_id}_{k}.pml"
    create_pymol_script(
        f"datasets/pdb_files/{protein_id}/{protein_id}_esmfold.pdb",
        f"datasets/docking/{protein_id}/protein.npy",
        output_script=output_script,
        k_clusters=k)
    replace_local_pathes(output_script)
