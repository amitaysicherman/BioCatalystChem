import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from collections import Counter
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from preprocessing.dock import get_reaction_attention_emd
from preprocessing.build_tokenizer import redo_ec_split

ec_to_filter = ['1.1.1.67',
                '1.1.1.3',
                '2.4.1.357',
                '2.3.1.65',
                '1.1.1.206',
                '2.3.1.115',
                '2.8.2.38',
                '1.1.1.292',
                '1.3.1.22']

# Load mappings
with open("datasets/docking/smiles_to_id.txt") as f:
    smiles_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}

ec_mapping = pd.read_csv("datasets/ec_map.csv")
ec_to_uniprot = defaultdict(str)
for i, row in ec_mapping.iterrows():
    ec_to_uniprot[row["EC_full"]] = row["Uniprot_id"]

# Read source lines
with open("datasets/ecreact/level4/src-train.txt") as f:
    src_lines = f.read().splitlines()[:-2]

# Process reactions
vecs = []
m_set = set()
ec_counter = Counter()
all_srcs = []
all_ecs = []
centers = {ec: None for ec in ec_to_filter}
for text in tqdm(src_lines):
    src, ec = redo_ec_split(text, True)
    if ec not in ec_to_filter:
        continue
    smiles = src.replace(" ", "")
    if smiles not in m_set:
        v = get_reaction_attention_emd(src, ec, ec_to_uniprot, smiles_to_id, alpha=0.5)
        m_set.add(smiles)
        if v is not None:
            vecs.append(v)
            all_srcs.append(smiles)
            all_ecs.append(ec)
            if centers[ec] is None:
                centers[ec] = get_reaction_attention_emd(src, ec, ec_to_uniprot, smiles_to_id, alpha=0)
        ec_counter[ec] += 1

# Filter and reduce dimensions
vecs = [x for x in vecs if x is not None]

vec_to_pca=vecs+[centers[ec] for ec in ec_to_filter]
vec_to_pca=np.array(vec_to_pca)
# apply T-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
all_vecs = tsne.fit_transform(vec_to_pca)
vecs=all_vecs[:len(vecs)]
centers={ec:all_vecs[i+len(vecs)] for i,ec in enumerate(ec_to_filter)}
# print(vec_to_pca.shape)
# pca = PCA(n_components=2)
# pca.fit(vec_to_pca)
# vecs=pca.transform(vecs)

# Create plot with molecule images
plt.figure(figsize=(10, 8))
def ec_to_color(ec):
    #use tab10 for 10 colors
    i=ec_to_filter.index(ec)
    return plt.cm.tab10(i)
colors=[ec_to_color(ec) for ec in all_ecs]
for single_ec in ec_to_filter:
    vec_in_ec=[vecs[i] for i in range(len(vecs)) if all_ecs[i]==single_ec]
    plt.scatter([x[0] for x in vec_in_ec],[x[1] for x in vec_in_ec],alpha=0.7,c=[ec_to_color(single_ec)],label=single_ec)
    center=centers[single_ec]
    # center_pca=pca.transform([center])
    center_pca=all_vecs[len(vecs)+ec_to_filter.index(single_ec)]
    plt.scatter(center_pca[0][0],center_pca[0][1],c=[ec_to_color(single_ec)],marker='x',s=100)

plt.legend()


def get_molecule_image(smiles, figsize=(2, 2)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            im = Draw.MolToImage(mol, size=(300, 300))
            return im
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
    return None


# Add molecule images precisely at points
# for i, txt in enumerate(all_srcs):
#     mol_img = get_molecule_image(txt)
#     if mol_img is not None:
#         # Create an OffsetImage
#         imagebox = OffsetImage(mol_img, zoom=0.5)
#         imagebox.image.axes = plt.gca()

#         # Create an annotation box at the exact point
#         ab = AnnotationBbox(imagebox, (vecs[i, 0], vecs[i, 1]),
#                             frameon=False,
#                             xycoords='data',
#                             boxcoords="offset points",
#                             pad=0)
#         plt.gca().add_artist(ab)

#         # # Optionally add SMILES text
#         # plt.annotate(txt, (vecs[i, 0], vecs[i, 1]),
#         #              xytext=(10, 10),
#         #              textcoords='offset points',
#         #              fontsize=8,
#         #              bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="gray", alpha=0.3))

plt.title("PCA of Reaction Embeddings with Molecule Structures")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.tight_layout()
plt.show()
