import pandas as pd


class SampleTags:
    def __init__(self, split):
        with open(f"datasets/ecreact/level4/src-{split}.txt", "r") as f:
            src_ec = f.read().splitlines()
        self.src = [x.split("|")[0].replace(" ", "") for x in src_ec]
        self.ec = [x.split("|")[1].strip() for x in src_ec]
        with open(f"datasets/ecreact/level4/tgt-{split}.txt", "r") as f:
            self.tgt = f.read().splitlines()
        self.tgt = [x.replace(" ", "") for x in self.tgt]
        with open(f"datasets/ecreact/level4/{split}_sources.txt", "r") as f:
            self.ds = f.read().splitlines()
        assert len(self.src) == len(self.tgt) == len(self.ec) == len(self.ds)
        self.df = pd.DataFrame({"src": self.src, "tgt": self.tgt, "ec": self.ec, "ds": self.ds})
        self.add_all_tag()

    def add_number_of_molecules(self):
        self.df["num_mol"] = self.df["src"].apply(lambda x: len(x.split(".")))

    def add_number_of_large_molecules(self, t=3):
        self.df["num_large_mol"] = self.df["src"].apply(lambda x: len([y for y in x.split(".") if len(y) > t]))

    def add_ec_l_1(self):
        self.df["ec_l_1"] = self.df["ec"].apply(lambda x: x.split(" ")[0].strip())

    def add_ec_l_2(self):
        self.df["ec_l_2"] = self.df["ec"].apply(lambda x: " ".join(x.split(" ")[:2]).strip())


    def add_most_common_molecules(self, n=10,len_threshold=3):
        all_molecules = []
        for x in self.df["src"]:

            all_molecules.extend([y for y in x.split(".") if len(y) > len_threshold])
        all_molecules = pd.Series(all_molecules)
        most_common_molecules = all_molecules.value_counts().head(n)
        for i in range(n):
            print(i,most_common_molecules.index[i], most_common_molecules[i])
            self.df[f"common_mol_{i}"] = self.df["src"].apply(lambda x: len([y for y in x.split(".") if y == most_common_molecules.index[i]]))

    def add_all_tag(self):
        self.add_number_of_molecules()
        self.add_number_of_large_molecules()
        self.add_ec_l_1()
        self.add_ec_l_2()
        self.add_most_common_molecules()


if __name__ == "__main__":
    test = SampleTags("test")
    print(len(test.df))
    print(test.df.head())
    print(test.df.columns)
