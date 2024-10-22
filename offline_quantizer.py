import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Sequence
from tqdm import tqdm
from preprocessing.build_tokenizer import redo_ec_split
from preprocessing.ec_to_vec import EC2Vec
from preprocessing.dock import get_reaction_attention_emd
from collections import defaultdict
import pandas as pd
from dataset import ECType
import pickle


class HierarchicalPCATokenizer:
    def __init__(self, n_hierarchical_clusters: Sequence[int] = (10, 50, 100, 500, 1000),
                 n_pca_components: int = 6, n_clusters_pca: int = 10):
        """
        Initialize tokenizer with hierarchical K-means and PCA-based clustering

        Args:
            n_hierarchical_levels: Number of hierarchical clustering levels
            clusters_per_level: Number of clusters at each hierarchical level
            n_pca_components: Number of PCA components to use
            n_clusters_pca: Number of clusters for each PCA dimension
        """
        self.n_hierarchical_clusters = n_hierarchical_clusters
        self.n_pca_components = n_pca_components
        self.n_clusters_pca = n_clusters_pca

        # Initialize models
        self.hierarchical_models = []
        self.pca_model = None
        self.pca_clusterers = []

    def fit(self, vectors: np.ndarray):
        """
        Fit the tokenizer to the vector dataset
        """
        # Fit PCA
        self.pca_model = PCA(n_components=self.n_pca_components)
        pca_vectors = self.pca_model.fit_transform(vectors)

        # Fit PCA dimension clusterers
        self.pca_clusters = []
        for dim in range(self.n_pca_components):
            dim_values = pca_vectors[:, dim].reshape(-1, 1)
            clusters = KMeans(n_clusters=self.n_clusters_pca, random_state=42)
            clusters.fit(dim_values)
            self.pca_clusters.append(clusters)

        # Fit hierarchical clusterers
        current_data = vectors
        self.hierarchical_models = []

        for k in self.n_hierarchical_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(current_data)
            self.hierarchical_models.append(kmeans)

    def tokenize_vector(self, vector: np.ndarray) -> List[str]:
        """
        Convert a single vector into a sequence of tokens
        """
        tokens = []
        current_vector = vector.reshape(1, -1)
        for level, model in enumerate(self.hierarchical_models):
            cluster = model.predict(current_vector)[0]
            tokens.append(f"H{level}-{cluster}")
        pca_vector = self.pca_model.transform(current_vector)
        for dim, clusterer in enumerate(self.pca_clusters):
            dim_value = pca_vector[:, dim].reshape(-1, 1)
            cluster = clusterer.predict(dim_value)[0]
            tokens.append(f"P{dim}-{cluster}")
        return tokens


def ec_to_filename(ec_type: ECType):
    return f"datasets/ecreact/vec_quant_{ec_type.value}.pkl"


def read_dataset_split(ec_type: ECType, split: str):
    input_base = "datasets/ecreact/level4"
    if ec_type == ECType.PRETRAINED:
        ec_to_vec = EC2Vec(load_model=False)
    if ec_type == ECType.DAE:
        with open("datasets/docking/smiles_to_id.txt") as f:
            smiles_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
        ec_mapping = pd.read_csv("datasets/ec_map.csv")
        ec_to_uniprot = defaultdict(str)
        for i, row in ec_mapping.iterrows():
            ec_to_uniprot[row["EC_full"]] = row["Uniprot_id"]
    with open(f"{input_base}/src-{split}.txt") as f:
        src_lines = f.read().splitlines()[:150]

    with open(f"{input_base}/tgt-{split}.txt") as f:
        tgt_lines = f.read().splitlines()[:150]

    assert len(src_lines) == len(tgt_lines)

    assert ec_type == ECType.PRETRAINED or ec_type == ECType.DAE
    src_ec = [redo_ec_split(text, True) for text in src_lines]
    src_lines = [x[0] for x in src_ec]
    ec_lines = [x[1] for x in src_ec]
    if ec_type == ECType.PRETRAINED:
        emb_lines = [ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]
    else:
        emb_lines = [
            get_reaction_attention_emd(text, ec, ec_to_uniprot, smiles_to_id)
            for text, ec in tqdm(zip(src_lines, ec_lines), total=len(src_lines))
        ]

    not_none_mask = [x is not None for x in emb_lines]
    src_lines = [src_lines[i] for i in range(len(src_lines)) if not_none_mask[i]]
    tgt_lines = [tgt_lines[i] for i in range(len(tgt_lines)) if not_none_mask[i]]
    emb_lines = [emb_lines[i] for i in range(len(emb_lines)) if not_none_mask[i]]
    return src_lines, tgt_lines, emb_lines


def train_model(ec_type: ECType):
    split = "train"
    _, _, emb_lines = read_dataset_split(ec_type, split)
    vecs = np.array(emb_lines)
    print(vecs.shape)
    tokenizer = HierarchicalPCATokenizer()
    tokenizer.fit(vecs)
    with open(ec_to_filename(ec_type), "wb") as f:
        pickle.dump(tokenizer, f)


def tokenize_dataset_split(ec_type: ECType, split):
    with open(ec_to_filename(ec_type), "rb") as f:
        tokenizer = pickle.load(f)
    src_lines, tgt_lines, emb_lines = read_dataset_split(ec_type, split)
    tokenized_lines = []
    for emb in tqdm(emb_lines):
        tokenized_lines.append(tokenizer.tokenize_vector(emb))
    assert len(src_lines) == len(tgt_lines) == len(tokenized_lines)
    output_base = f"datasets/ecreact/quant_{ec_type.value}"
    os.makedirs(output_base, exist_ok=True)
    src_out = open(f"{output_base}/src-{split}.txt", "w")
    tgt_out = open(f"{output_base}/tgt-{split}.txt", "w")
    for s, t, e in zip(src_lines, tgt_lines, tokenized_lines):
        s = s + " | " + " ".join(e)
        src_out.write(s + "\n")
        tgt_out.write(t + "\n")
    src_out.close()
    tgt_out.close()


if __name__ == "__main__":
    for ec_type in [ECType.DAE,ECType.PRETRAINED]:
        train_model(ec_type)
        for split in ["train", "valid", "test"]:
            tokenize_dataset_split(ec_type, split)
