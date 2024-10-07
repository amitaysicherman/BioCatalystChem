import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from preprocessing.build_tokenizer import redo_ec_split, encode_eos_pad
from utils import remove_ec
from preprocessing.ec_to_vec import EC2Vec
from preprocessing.dock import get_reaction_attention_emd
from enum import Enum
from collections import defaultdict
import pandas as pd
import random

class ECType(Enum):
    NO_EC = 0
    PAPER = 1
    PRETRAINED = 2
    DAE = 3

DEFAULT_EMB_VALUE = torch.tensor([0]*2560)

def get_ec_type(use_ec, ec_split, dae):
    if dae:
        return ECType.DAE
    if not use_ec:
        return ECType.NO_EC
    if ec_split:
        return ECType.PAPER
    return ECType.PRETRAINED


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, weights=None, max_length=200, DEBUG=False,
                 ec_type=ECType.NO_EC):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = []
        self.DEBUG = DEBUG
        self.ec_type = ec_type
        self.lookup_embeddings = []
        if ec_type == ECType.PRETRAINED:
            self.ec_to_vec = EC2Vec(load_model=False)
        if ec_type == ECType.DAE:
            with open("datasets/docking/smiles_to_id.txt") as f:
                self.smiles_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
            ec_mapping = pd.read_csv("datasets/ec_map.csv")
            self.ec_to_uniprot = defaultdict(str)
            for i, row in ec_mapping.iterrows():
                self.ec_to_uniprot[row["EC_full"]] = row["Uniprot_id"]

        if weights is None:
            weights = [1] * len(datasets)
        else:
            assert len(weights) == len(datasets)
        for ds, w in zip(datasets, weights):
            self.load_dataset(f"datasets/{ds}", split, w,have_ec="ec" in ds)
        random.shuffle(self.data)

    def load_dataset(self, input_base, split, w,have_ec=True):
        with open(f"{input_base}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        with open(f"{input_base}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        assert len(src_lines) == len(tgt_lines)
        emb_lines = [DEFAULT_EMB_VALUE] * len(src_lines)
        if have_ec:
            if self.ec_type == ECType.NO_EC:
                src_lines = [remove_ec(text) for text in src_lines]
                tgt_lines = [remove_ec(text) for text in tgt_lines]
            elif self.ec_type == ECType.PRETRAINED or self.ec_type == ECType.DAE:
                src_ec = [redo_ec_split(text, True) for text in src_lines]
                src_lines = [x[0] for x in src_ec]
                ec_lines = [x[1] for x in src_ec]
                if self.ec_type == ECType.PRETRAINED:
                    emb_lines = [self.ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]

                else:
                    emb_lines = [
                        get_reaction_attention_emd(text, ec, self.ec_to_uniprot, self.smiles_to_id)
                        for text, ec in tqdm(zip(src_lines, ec_lines), total=len(src_lines))
                    ]

                not_none_mask = [x is not None for x in emb_lines]
                len_before = len(src_lines)
                src_lines = [src_lines[i] for i in range(len(src_lines)) if not_none_mask[i]]
                tgt_lines = [tgt_lines[i] for i in range(len(tgt_lines)) if not_none_mask[i]]
                emb_lines = [emb_lines[i] for i in range(len(emb_lines)) if not_none_mask[i]]
                len_after = len(src_lines)
                print(f"Removed {len_before - len_after} samples, total: {len_after}, {len_before}")

        if self.DEBUG:
            src_lines = src_lines[:1]
            tgt_lines = tgt_lines[:1]
            emb_lines = emb_lines[:1]

        skip_count = 0
        data = []
        for i in tqdm(range(len(src_lines))):
            input_id, attention_mask = encode_eos_pad(self.tokenizer, src_lines[i], self.max_length)
            label, label_mask = encode_eos_pad(self.tokenizer, tgt_lines[i], self.max_length)
            if input_id is None or label is None:
                skip_count += 1
                continue
            label[label_mask == 0] = -100
            emb = emb_lines[i]
            data.append((input_id, attention_mask, label, emb))
        for _ in range(w):
            self.data.extend(data)

    def __len__(self):
        return len(self.data)

    def data_to_dict(self, data):
        return {"input_ids": data[0], "attention_mask": data[1], "labels": data[2], "emb": data[3]}

    def __getitem__(self, idx):
        return self.data_to_dict(self.data[idx])
