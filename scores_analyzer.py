import os

import pandas as pd
from tqdm import tqdm

from preprocessing.sample_tagging import SampleTags


def get_scores_from_df(df, indexes):
    if indexes is not None:
        indexes_in_df = [x for x in df.index if x in indexes]
        df = df.loc[indexes_in_df]
    return df["res"].mean()


def read_file(file_path):
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
        self.run_name = run_name
        self.base_dir = f"results/{run_name}"

        all_files = [x for x in os.listdir(self.base_dir) if x.endswith(".txt")]
        all_split_n_k = [file_name_to_split_n_k(x) for x in all_files]
        valid_n_k = [(n, k) for split, n, k in all_split_n_k if split == "valid"]
        test_n_k = [(n, k) for split, n, k in all_split_n_k if split == "test"]
        n_k = set(valid_n_k).intersection(test_n_k)
        self.configs = n_k
        self.valid_config_to_df = {c: read_file(os.path.join(self.base_dir, config_to_file_name("valid", c[0], c[1])))
                                   for c in n_k}
        self.test_config_to_df = {c: read_file(os.path.join(self.base_dir, config_to_file_name("test", c[0], c[1]))) for
                                  c in n_k}

    def get_k_configs(self, k):
        return [c for c in self.configs if c[1] == k]

    def get_score_k(self, k, index_valid, index_test):
        valid_scores = []
        test_scores = []
        configs = self.get_k_configs(k)
        for c in configs:
            valid_scores.append(get_scores_from_df(self.valid_config_to_df[c], index_valid))
            test_scores.append(get_scores_from_df(self.test_config_to_df[c], index_test))
        best_valid = max(valid_scores)
        best_test = test_scores[valid_scores.index(best_valid)]
        return best_test


def impact_score(results: pd.Series):
    base_lines_indexes = [x for x in results.index if "paper" in x or "regular" in x]
    pre_indexes = [x for x in results.index if "pre" in x]
    dae = [x for x in results.index if "dae" in x]
    base_score = results.loc[base_lines_indexes].max()
    pre_score = results.loc[pre_indexes].max()
    dae_score = results.loc[dae].max()
    return dae_score - max(base_score, pre_score), max(dae_score, pre_score) - base_score, dae_score - base_score


all_methods = os.listdir("results")
all_methods = [x for x in all_methods if os.path.isdir(os.path.join("results", x))]
print(all_methods)
run_num_to_obj = dict()
for name in tqdm(all_methods):
    if name == "full":
        continue
    if len([x for x in os.listdir(os.path.join("results", name)) if x.endswith(".txt")]) == 0:
        continue
    run_num_to_obj[name] = ValTestFiles(name)

sample_tags_valid = SampleTags(split="valid")
sample_tags_test = SampleTags(split="test", common_molecules=sample_tags_valid.common_molecules,common_ec=sample_tags_valid.common_ec)
datasets = [(), ("ds", lambda x: x == "brenda_reaction_smiles"), ("ds", lambda x: x == "metanetx_reaction_smiles"),
            ("ds", lambda x: x == "pathbank_reaction_smiles"), ("ds", lambda x: x == "rhea_reaction_smiles")]
datasets_names = ["all", "brenda", "metanetx", "pathbank", "rhea"]
singaltons = [(), ("num_train_tgt", lambda x: x == 0), ("num_train_src", lambda x: x == 0),
              ("num_train_ec_3", lambda x: x == 0), ("num_train_ec_4", lambda x: x == 0),
              ("num_train_src", lambda x: x > 0)]
singaltons_names = ["all", "no_tgt", "no_src", "no_ec3", "no_ec4", "train_src"]
reaction_lengths = [(), ("num_mol", lambda x: x > 1), ("num_large_mol", lambda x: x > 1), ("num_mol", lambda x: x > 2),
                    ("num_large_mol", lambda x: x > 2), ("num_mol", lambda x: x > 3),
                    ("num_large_mol", lambda x: x > 3)]
length_names = ["all", "more_than_1", "large_more_than_1", "more_than_2", "large_more_than_2", "more_than_3",
                "large_more_than_3"]
filter_categories = ["dataset", "singalton", "length"]
filter_configs = []
for dataset_name, dataset_filter in zip(datasets_names, datasets):
    for singalton_name, singalton_filter in zip(singaltons_names, singaltons):
        for length_name, length_filter in zip(length_names, reaction_lengths):
            names = [dataset_name, singalton_name, length_name]
            configs = [dataset_filter, singalton_filter, length_filter]
            remove_index = [i for i in range(len(names)) if configs[i] == ()]
            names = [names[i] if i not in remove_index else "all" for i in range(len(names))]
            configs = [configs[i] for i in range(len(configs)) if i not in remove_index]
            filter_configs.append((names, configs))

new_names = [f"common_mol_{i}" for i in range(10)] + [f"common_ec_{i}" for i in range(10)]
new_configs = [("common_mol_0", lambda x: x > 0), ("common_mol_1", lambda x: x > 0), ("common_mol_2", lambda x: x > 0),
               ("common_mol_3", lambda x: x > 0), ("common_mol_4", lambda x: x > 0), ("common_mol_5", lambda x: x > 0),
               ("common_mol_6", lambda x: x > 0), ("common_mol_7", lambda x: x > 0), ("common_mol_8", lambda x: x > 0),
               ("common_mol_9", lambda x: x > 0), ("common_ec_0", lambda x: x > 0), ("common_ec_1", lambda x: x > 0),
               ("common_ec_2", lambda x: x > 0), ("common_ec_3", lambda x: x > 0), ("common_ec_4", lambda x: x > 0),
               ("common_ec_5", lambda x: x > 0), ("common_ec_6", lambda x: x > 0), ("common_ec_7", lambda x: x > 0),
               ("common_ec_8", lambda x: x > 0), ("common_ec_9", lambda x: x > 0)]
for i in range(len(new_names)):
    filter_configs.append(([new_names[i]], [new_configs[i]]))

all_results = []
for conf_name, filter_tags in tqdm(filter_configs):
    print(conf_name)
    index_valid = sample_tags_valid.get_query_indexes(filter_tags)
    index_test = sample_tags_test.get_query_indexes(filter_tags)
    config_results = dict()
    for name, obj in run_num_to_obj.items():
        config_results[name] = obj.get_score_k(5, index_valid, index_test)
    config_results['count'] = len(index_test)
    for i in range(len(conf_name)):
        config_results[f'f_{filter_categories[i]}'] = conf_name[i]

    i1, i2, i3 = impact_score(pd.Series(config_results))
    config_results['impact1'] = i1
    config_results['impact2'] = i2
    config_results['impact3'] = i3
    results = pd.Series(config_results)
    all_results.append(results)
print(all_results)
# convert to dataframe
df = pd.DataFrame(all_results)
print(df)
df.to_csv("results/impact_results.csv")
