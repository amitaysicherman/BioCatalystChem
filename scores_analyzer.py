import os
import pandas as pd

from preprocessing.sample_tagging import SampleTags
from dataclasses import dataclass


def read_res_file(file_path):
    return pd.read_csv(file_path, index_col=0, header=None, names=["index", "res"],
                       dtype={"res": int, "index": int})


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

    def get_score_k(self, k):
        valid_scores = []
        test_scores = []
        configs = self.get_k_configs(k)
        for c in configs:
            n, k = c
            valid_file = config_to_file_name("valid", n, k)
            valid_scores.append(read_res_file(os.path.join(self.base_dir, valid_file))["res"].mean())
            test_file = config_to_file_name("test", n, k)
            test_scores.append(read_res_file(os.path.join(self.base_dir, test_file))["res"].mean())
        best_valid = max(valid_scores)
        best_test = test_scores[valid_scores.index(best_valid)]
        return best_test

    def get_score(self):
        return {k: self.get_score_k(k) for k in self.all_k}


all_methods = os.listdir("results")
all_methods = [x for x in all_methods if os.path.isdir(os.path.join("results", x))]
print(all_methods)
all_results = []
for name in all_methods:
    if len([x for x in os.listdir(os.path.join("results", name)) if x.endswith(".txt")]) == 0:
        continue
    res = ValTestFiles(name).get_score()

    res["name"] = name
    all_results.append(res)
results = pd.DataFrame(all_results)
print(results)
# all_df = []
# for file in scores_files:
#     file_path = os.path.join(base_dir, file)
#     df = read_res_file(file_path)
#     df = df[~df.index.duplicated(keep='first')]
#     df['name'] = file
#     all_df.append(df)

# all_df = pd.concat(all_df)
# all_df = all_df.reset_index().pivot(index='index', columns="name", values="res").dropna()  # to_csv("tmp.csv")
# print(all_df)

3 / 0

test = SampleTags("test")
print(len(test.df))

base_dir = "results/full"
scores_files = [x for x in os.listdir(base_dir) if x.endswith(".csv")]
print(scores_files)
all_df = []
for file in scores_files:
    file_path = os.path.join(base_dir, file)
    df = read_res_file(file_path)
    df = df[~df.index.duplicated(keep='first')]

    df['name'] = file
    all_df.append(df)
all_df = pd.concat(all_df)
all_df = all_df.reset_index().pivot(index='index', columns="name", values="res").dropna()  # to_csv("tmp.csv")
all_df.merge(test.df, left_index=True, right_index=True, how="left").dropna().to_csv("tmp.csv")
