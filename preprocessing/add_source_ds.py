import os
from os.path import join as pjoin
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class ReactionToSource:

    def __init__(self, csv_file="datasets/ecreact/ecreact-1.0.csv"):
        self.csv = pd.read_csv(csv_file)
        self.reaction_to_source = {}
        for i, row in tqdm(self.csv.iterrows(),total=len(self.csv)):
            src_ec, tgt = row["rxn_smiles"].split(">>")
            src, ec = src_ec.split("|")
            src = src.strip().replace(" ", "")
            src = self.organize_mols(src)
            tgt = tgt.strip().replace(" ", "")
            tgt = self.organize_mols(tgt)
            reaction = f"{src}|{ec}>>{tgt}"
            if reaction in self.reaction_to_source:
                print(f"Duplicate reaction {reaction}")
            self.reaction_to_source[reaction] = row["source"]

    def to_canonical(self, s):
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m)

    def organize_mols(self, s):
        s = s.split(".")
        s = sorted(s)

        s = ".".join([self.to_canonical(x) for x in s])
        return s

    def get_source(self, src, tgt, ec=None):
        if ec is None:
            if "|" not in src:
                print("No ec in src")
                return ""
            src, *ec = src.split("|")
            ec = ec[0]
        src = src.strip().replace(" ", "")
        src = self.organize_mols(src)
        tgt = tgt.strip().replace(" ", "")
        tgt = self.organize_mols(tgt)
        ec = ".".join([x.replace("[", "").replace("]", "")[1:] for x in ec.strip().split(" ")])
        reaction = f"{src}|{ec}>>{tgt}"
        if reaction not in self.reaction_to_source:
            print(f"Reaction {reaction} not found in csv")
            return ""
        return self.reaction_to_source[reaction]

    def get_source_to_ds_split(self, base_dir, split):
        src_file = pjoin(base_dir, f"src-{split}.txt")
        tgt_file = pjoin(base_dir, f"tgt-{split}.txt")
        with open(src_file) as f:
            src_lines = f.readlines()
        with open(tgt_file) as f:
            tgt_lines = f.readlines()
        sources = []
        for src, tgt in zip(src_lines, tgt_lines):
            sources.append(self.get_source(src, tgt))
        return sources

    def get_source_to_ds(self, base_dir):
        train_src = self.get_source_to_ds_split(base_dir, "train")
        valid_src = self.get_source_to_ds_split(base_dir, "valid")
        test_src = self.get_source_to_ds_split(base_dir, "test")
        return train_src, valid_src, test_src


reaction_to_source = ReactionToSource()
base_dir = pjoin('datasets', 'ecreact')
# get all subdirectories
subdirs = [pjoin(base_dir, x) for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
for subdir in tqdm(subdirs):
    if "level4" in subdir or "quant" in subdir:
        train_src, valid_src, test_src = reaction_to_source.get_source_to_ds(subdir)
        with open(pjoin(subdir, "train_sources.txt"), "w") as f:
            for src in train_src:
                f.write(f"{src}\n")
        with open(pjoin(subdir, "valid_sources.txt"), "w") as f:
            for src in valid_src:
                f.write(f"{src}\n")
        with open(pjoin(subdir, "test_sources.txt"), "w") as f:
            for src in test_src:
                f.write(f"{src}\n")
