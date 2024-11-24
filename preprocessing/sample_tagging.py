import pandas as pd


def load_df(split):
    with open(f"datasets/ecreact/level4/src-{split}.txt", "r") as f:
        src_ec = f.read().splitlines()
    src = [x.split("|")[0].replace(" ", "") for x in src_ec]
    ec = [x.split("|")[1].strip() for x in src_ec]
    with open(f"datasets/ecreact/level4/tgt-{split}.txt", "r") as f:
        tgt = f.read().splitlines()
    tgt = [x.replace(" ", "") for x in tgt]
    with open(f"datasets/ecreact/level4/{split}_sources.txt", "r") as f:
        ds = f.read().splitlines()
    assert len(src) == len(tgt) == len(ec) == len(ds)
    df = pd.DataFrame({"src": src, "tgt": tgt, "ec": ec, "ds": ds})
    return df


def ec_to_level(ec, level=1):
    return " ".join(ec.split(" ")[:level])


class SampleTags:
    def __init__(self, split):
        self.df = load_df(split)
        self.train_df = load_df("train")
        self.add_all_tag()

    def add_number_of_molecules(self):
        self.df["num_mol"] = self.df["src"].apply(lambda x: len(x.split(".")))

    def add_number_of_large_molecules(self, t=3):
        self.df["num_large_mol"] = self.df["src"].apply(lambda x: len([y for y in x.split(".") if len(y) > t]))

    def add_ec_level(self, level):
        self.df[f"ec_l_{level}"] = self.df["ec"].apply(lambda x: ec_to_level(x, level))

    def add_most_common_molecules(self, n=10, len_threshold=3):
        all_molecules = []
        for x in self.df["src"]:
            all_molecules.extend([y for y in x.split(".") if len(y) > len_threshold])
        all_molecules = pd.Series(all_molecules)
        most_common_molecules = all_molecules.value_counts().head(n)
        for i in range(n):
            print(i, most_common_molecules.index[i], most_common_molecules[i])
            self.df[f"common_mol_{i}"] = self.df["src"].apply(
                lambda x: len([y for y in x.split(".") if y == most_common_molecules.index[i]]))

    def add_num_train_ec(self, level=1):
        train_ec = self.train_df[f"ec"].apply(lambda x: ec_to_level(x, level)).value_counts()
        self.df[f"num_train_ec_{level}"] = self.df["ec"].apply(lambda x: train_ec.get(x, 0))

    def add_num_train_src(self):
        train_src = self.train_df["src"].value_counts()
        self.df["num_train_src"] = self.df["src"].apply(lambda x: train_src.get(x, 0))

    def add_all_tag(self):
        self.add_number_of_molecules()
        self.add_number_of_large_molecules()
        self.add_ec_level(1)
        self.add_ec_level(2)
        self.add_most_common_molecules()
        self.add_num_train_ec(1)
        self.add_num_train_ec(2)
        self.add_num_train_src()


if __name__ == "__main__":
    test = SampleTags("test")
    print(len(test.df))
    print(test.df.head())
    print(test.df.columns)
