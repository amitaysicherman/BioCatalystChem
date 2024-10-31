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


# from multiprocessing import Pool


class HierarchicalPCATokenizer:
    def __init__(self, n_hierarchical_clusters: int = 5,
                 n_pca_components: int = 6, n_clusters_pca: int = 10):
        """
        Initialize tokenizer with hierarchical K-means and PCA-based clustering

        Args:
            n_hierarchical_levels: Number of hierarchical clustering levels
            clusters_per_level: Number of clusters at each hierarchical level
            n_pca_components: Number of PCA components to use
            n_clusters_pca: Number of clusters for each PCA dimension
        """
        n_to_val_hierarchy = {
            0: [],
            1: [10],
            2: [10, 50],
            3: [10, 50, 100],
            4: [10, 50, 100, 250],
            5: [10, 50, 100, 250, 500],
            6: [10, 50, 100, 250, 500, 1000],
            7: [10, 50, 100, 250, 500, 1000, 1500],
        }
        self.vals_hierarchical_clusters = n_to_val_hierarchy[n_hierarchical_clusters]
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

        for k in self.vals_hierarchical_clusters:
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

    def get_all_tokens(self):
        tokens = []
        for level, k in enumerate(self.vals_hierarchical_clusters):
            for cluster in range(k):
                tokens.append(f"H{level}-{cluster}")
        for dim in range(self.n_pca_components):
            for cluster in range(self.n_clusters_pca):
                tokens.append(f"P{dim}-{cluster}")
        return tokens


def args_to_quant_model_file(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca):
    return f"datasets/ecreact/vec_quant_{ec_type.value}_{n_hierarchical_clusters}_{n_pca_components}_{n_clusters_pca}.pkl"


def args_to_quant_dataset(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha):
    ec_type = ec_type.value if ec_type != ECType.DAE else f"{ec_type.value}-{alpha}"
    return f"datasets/ecreact/quant_{ec_type}_{n_hierarchical_clusters}_{n_pca_components}_{n_clusters_pca}"


# def get_reaction_attention_emb_wrapper(args):
#     text, ec, ec_to_uniprot, smiles_to_id, alpha = args
#     return get_reaction_attention_emd(text, ec, ec_to_uniprot, smiles_to_id, alpha=alpha)


def read_dataset_split(ec_type: ECType, split: str, alpha):
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
        src_lines = f.read().splitlines()

    with open(f"{input_base}/tgt-{split}.txt") as f:
        tgt_lines = f.read().splitlines()

    assert len(src_lines) == len(tgt_lines)

    assert ec_type == ECType.PRETRAINED or ec_type == ECType.DAE
    src_ec = [redo_ec_split(text, True) for text in src_lines]
    src_lines = [x[0] for x in src_ec]
    ec_lines = [x[1] for x in src_ec]

    if ec_type == ECType.PRETRAINED:
        emb_lines = [ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]
    else:
        emb_lines = [
            get_reaction_attention_emd(text, ec, ec_to_uniprot, smiles_to_id, alpha=alpha)
            for text, ec in tqdm(zip(src_lines, ec_lines), total=len(src_lines))
        ]

    not_none_mask = [x is not None for x in emb_lines]
    src_lines = [src_lines[i] for i in range(len(src_lines)) if not_none_mask[i]]
    tgt_lines = [tgt_lines[i] for i in range(len(tgt_lines)) if not_none_mask[i]]
    emb_lines = [emb_lines[i] for i in range(len(emb_lines)) if not_none_mask[i]]
    return src_lines, tgt_lines, emb_lines


def train_model(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha):
    split = "train"
    outputfile = args_to_quant_model_file(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca)
    if os.path.exists(outputfile):
        print("Model already exists")
        return
    _, _, emb_lines = read_dataset_split(ec_type, split, alpha)
    vecs = np.array(emb_lines)
    print(vecs.shape)
    tokenizer = HierarchicalPCATokenizer(n_hierarchical_clusters, n_pca_components, n_clusters_pca)
    tokenizer.fit(vecs)
    with open(args_to_quant_model_file(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca), "wb") as f:
        pickle.dump(tokenizer, f)


def tokenize_dataset_split(ec_type: ECType, split, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha):
    with open(args_to_quant_model_file(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca), "rb") as f:
        tokenizer: HierarchicalPCATokenizer = pickle.load(f)
    src_lines, tgt_lines, emb_lines = read_dataset_split(ec_type, split, alpha=alpha)
    tokenized_lines = [
        tokenizer.tokenize_vector(e) for e in emb_lines
    ]
    assert len(src_lines) == len(tgt_lines) == len(tokenized_lines)

    output_base = args_to_quant_dataset(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha)

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hierarchical_clusters", type=int, default=5)
    parser.add_argument("--n_pca_components", type=int, default=6)
    parser.add_argument("--n_clusters_pca", type=int, default=10)
    parser.add_argument("--alpha", type=int, default=50)
    parser.add_argument("--ec_type", type=int, default=ECType.PRETRAINED.value)
    args = parser.parse_args()
    args.alpha = float(args.alpha / 100)
    if args.ec_type == ECType.PRETRAINED.value:
        ec_type = ECType.PRETRAINED
    elif args.ec_type == ECType.DAE.value:
        ec_type = ECType.DAE
    else:
        raise ValueError("Invalid ec_type")
    train_model(ec_type, args.n_hierarchical_clusters, args.n_pca_components, args.n_clusters_pca, args.alpha)
    for split in ["train", "valid", "test"]:
        tokenize_dataset_split(ec_type, split, args.n_hierarchical_clusters, args.n_pca_components,
                               args.n_clusters_pca, alpha=args.alpha)
