import os
import pandas as pd
from tqdm import tqdm
from preprocessing.sample_tagging import SampleTags
from dataclasses import dataclass


def get_scores_from_file(file_path, indexes):
    df = pd.read_csv(file_path, index_col=0, header=None, names=["index", "res"],
                     dtype={"res": int, "index": int})
    print(len(df))
    if indexes is not None:
        indexes_in_df = [x for x in df.index if x in indexes]
        df = df.loc[indexes_in_df]
    print(len(df))
    return df["res"].mean()


def file_name_to_split_n_k(file_name):
    split, n, k = file_name.split("_")
    k = k.split(".")[0].replace("k", "")
    return split, n, int(k)


def config_to_file_name(split, n, k):
    return f"{split}_{n}_k{k}.txt"


class ValTestFiles:
    def __init__(self, run_name):
        self.base_dir = f"results/{run_name}"

        all_files = [x for x in os.listdir(self.base_dir) if x.endswith(".txt")]
        all_split_n_k = [file_name_to_split_n_k(x) for x in all_files]
        valid_n_k = [(n, k) for split, n, k in all_split_n_k if split == "valid"]
        test_n_k = [(n, k) for split, n, k in all_split_n_k if split == "test"]
        n_k = set(valid_n_k).intersection(test_n_k)
        self.configs = n_k
        self.all_k = set([k for n, k in n_k])

    def get_k_configs(self, k):
        return [c for c in self.configs if c[1] == k]

    def get_score_k(self, k, index_valid, index_test):
        valid_scores = []
        test_scores = []
        configs = self.get_k_configs(k)
        for c in configs:
            n, k = c
            valid_file = config_to_file_name("valid", n, k)
            valid_scores.append(get_scores_from_file(os.path.join(self.base_dir, valid_file), index_valid))
            test_file = config_to_file_name("test", n, k)
            test_scores.append(get_scores_from_file(os.path.join(self.base_dir, test_file), index_test))
        best_valid = max(valid_scores)
        best_test = test_scores[valid_scores.index(best_valid)]
        return best_test

    def get_score(self, index_valid, index_test):
        return {k: self.get_score_k(k, index_valid, index_test) for k in self.all_k}


sample_tags_valid = SampleTags(split="valid")
sample_tags_test = SampleTags(split="test")

filter_tags = [("ds", lambda x: x == "rhea_reaction_smiles"),("num_train_tgt", lambda x: x == 0)]
index_valid = sample_tags_valid.get_query_indexes(filter_tags)
index_test = sample_tags_test.get_query_indexes(filter_tags)
print(len(index_valid), len(index_test))
all_methods = os.listdir("results")
all_methods = [x for x in all_methods if os.path.isdir(os.path.join("results", x))]
print(all_methods)
all_results = dict()
for name in tqdm(all_methods):
    if name == "full":
        continue
    if len([x for x in os.listdir(os.path.join("results", name)) if x.endswith(".txt")]) == 0:
        continue
    res = ValTestFiles(name).get_score_k(5, index_valid, index_test)

    all_results[name] = res
    # all_results.append(res)
print(all_results)
results = pd.Series(all_results)
print(results)
