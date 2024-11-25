import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List
from tqdm import tqdm
from preprocessing.build_tokenizer import redo_ec_split
from preprocessing.ec_to_vec import EC2Vec
# from preprocessing.dock import get_reaction_attention_emd
from preprocessing.dock import Docker

from collections import defaultdict
import pandas as pd
from dataset import ECType
import pickle
from concurrent.futures import ProcessPoolExecutor

n_cpu = os.cpu_count()


class ResidualPCATokenizer:
    def __init__(self, n_residual_clusters: int = 5,
                 n_pca_components: int = 6, n_clusters_pca: int = 10):
        """
        Initialize tokenizer with residual K-means and PCA-based clustering

        Args:
            n_residual_clusters: Number of residual clustering levels
            n_pca_components: Number of PCA components to use
            n_clusters_pca: Number of clusters for each PCA dimension
        """
        self.n_residual_clusters = n_residual_clusters
        self.n_pca_components = n_pca_components
        self.n_clusters_pca = n_clusters_pca

        # Initialize models
        self.residual_models = []
        self.pca_model = None
        self.pca_clusterers = []

    def fit(self, vectors: np.ndarray):
        """
        Fit the tokenizer to the vector dataset using residual clustering
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

        # Fit residual clusterers
        current_data = vectors.copy()
        self.residual_models = []

        for _ in range(self.n_residual_clusters):
            kmeans = KMeans(n_clusters=self.n_clusters_pca, random_state=42)
            kmeans.fit(current_data)
            self.residual_models.append(kmeans)
            closest_centroids = kmeans.cluster_centers_[kmeans.labels_]
            current_data = current_data - closest_centroids

    def tokenize_vector(self, vector: np.ndarray) -> List[str]:
        """
        Convert a single vector into a sequence of tokens using residual clustering
        """
        tokens = []
        current_vector = vector.copy().reshape(1, -1)

        # Get residual cluster tokens
        for level, model in enumerate(self.residual_models):
            cluster = model.predict(current_vector)[0]
            tokens.append(f"R{level}-{cluster}")
            # Calculate residual for next iteration
            centroid = model.cluster_centers_[cluster]
            current_vector = current_vector - centroid.reshape(1, -1)

        # Get PCA tokens
        pca_vector = self.pca_model.transform(vector.reshape(1, -1))
        for dim, clusterer in enumerate(self.pca_clusters):
            dim_value = pca_vector[:, dim].reshape(-1, 1)
            cluster = clusterer.predict(dim_value)[0]
            tokens.append(f"P{dim}-{cluster}")

        return tokens

    def get_all_tokens(self):
        tokens = []
        for level in range(self.n_residual_clusters):
            for cluster in range(self.n_clusters_pca):
                tokens.append(f"R{level}-{cluster}")
        for dim in range(self.n_pca_components):
            for cluster in range(self.n_clusters_pca):
                tokens.append(f"P{dim}-{cluster}")
        return tokens


def get_args_name(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2):
    if ec_type == ECType.PRETRAINED:
        ec_type = ec_type.value
    elif ec_type == ECType.DAE:
        ec_type = f"{ec_type.value}-{alpha}"
        if daev2:
            ec_type += "-v2"
    else:
        raise ValueError("Invalid ec_type", ec_type)
    return f"{ec_type}_{n_hierarchical_clusters}_{n_pca_components}_{n_clusters_pca}"


def args_to_quant_model_file(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2):
    name = get_args_name(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2)
    return f"datasets/ecreact/vec_quant_{name}.pkl"


def args_to_quant_dataset(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2):
    name = get_args_name(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2)
    return f"datasets/ecreact/quant_{name}"


def read_dataset_split(ec_type: ECType, split: str, alpha, daev2):
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
    if os.path.exists(f"{input_base}/{split}_sources.txt"):
        with open(f"{input_base}/{split}_sources.txt") as f:
            source_lines = f.read().splitlines()
    else:
        source_lines = [0] * len(src_lines)
    assert len(src_lines) == len(tgt_lines) == len(source_lines)

    assert ec_type == ECType.PRETRAINED or ec_type == ECType.DAE
    src_ec = [redo_ec_split(text, True) for text in src_lines]
    src_lines = [x[0] for x in src_ec]
    ec_lines = [x[1] for x in src_ec]

    if ec_type == ECType.PRETRAINED:
        emb_lines = [ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]
    else:
        doker = Docker(alpha, daev2)
        emb_lines = [doker.dock_src_ec(src, ec) for src, ec in zip(src_lines, ec_lines)]

    not_none_mask = [x is not None for x in emb_lines]
    src_lines = [src_lines[i] for i in range(len(src_lines)) if not_none_mask[i]]
    tgt_lines = [tgt_lines[i] for i in range(len(tgt_lines)) if not_none_mask[i]]
    emb_lines = [emb_lines[i] for i in range(len(emb_lines)) if not_none_mask[i]]
    source_lines = [source_lines[i] for i in range(len(source_lines)) if not_none_mask[i]]
    return src_lines, tgt_lines, emb_lines, source_lines


def train_model(ec_type: ECType, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2):
    split = "train"
    outputfile = args_to_quant_model_file(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha,
                                          daev2)
    if os.path.exists(outputfile):
        print("Model already exists")
    _, _, emb_lines, _ = read_dataset_split(ec_type, split, alpha, daev2)
    vecs = np.array(emb_lines)
    print(vecs.shape)
    tokenizer = ResidualPCATokenizer(n_hierarchical_clusters, n_pca_components, n_clusters_pca)
    tokenizer.fit(vecs)
    with open(
            args_to_quant_model_file(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2),
            "wb") as f:
        pickle.dump(tokenizer, f)


def tokenize_dataset_split(ec_type: ECType, split, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha,
                           daev2):
    with open(
            args_to_quant_model_file(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, daev2),
            "rb") as f:
        tokenizer: ResidualPCATokenizer = pickle.load(f)
    src_lines, tgt_lines, emb_lines, source_lines = read_dataset_split(ec_type, split, alpha=alpha, daev2=daev2)
    tokenized_lines = [
        tokenizer.tokenize_vector(e) for e in emb_lines
    ]
    assert len(src_lines) == len(tgt_lines) == len(tokenized_lines)

    output_base = args_to_quant_dataset(ec_type, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha,
                                        daev2)

    os.makedirs(output_base, exist_ok=True)
    src_out = open(f"{output_base}/src-{split}.txt", "w")
    tgt_out = open(f"{output_base}/tgt-{split}.txt", "w")
    for s, t, e in zip(src_lines, tgt_lines, tokenized_lines):
        s = s + " | " + " ".join(e)
        src_out.write(s + "\n")
        tgt_out.write(t + "\n")
    src_out.close()
    tgt_out.close()
    with open(f"{output_base}/{split}_sources.txt", "w") as f:
        for source in source_lines:
            f.write(f"{source}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hierarchical_clusters", type=int, default=5)
    parser.add_argument("--n_pca_components", type=int, default=6)
    parser.add_argument("--n_clusters_pca", type=int, default=10)
    parser.add_argument("--alpha", type=int, default=50)
    parser.add_argument("--ec_type", type=int, default=ECType.PRETRAINED.value)
    parser.add_argument("--daev2", type=int, default=0)
    args = parser.parse_args()
    args.alpha = float(args.alpha / 100)
    if args.ec_type == ECType.PRETRAINED.value:
        ec_type = ECType.PRETRAINED
    elif args.ec_type == ECType.DAE.value:
        ec_type = ECType.DAE
    else:
        raise ValueError("Invalid ec_type")
    train_model(ec_type, args.n_hierarchical_clusters, args.n_pca_components, args.n_clusters_pca, args.alpha,
                args.daev2)
    for split in ["train", "valid", "test"]:
        tokenize_dataset_split(ec_type, split, args.n_hierarchical_clusters, args.n_pca_components,
                               args.n_clusters_pca, alpha=args.alpha, daev2=args.daev2)
