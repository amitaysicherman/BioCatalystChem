import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from preprocessing.build_tokenizer import redo_ec_split, encode_eos_pad
from utils import remove_ec
from preprocessing.ec_to_vec import EC2Vec
from preprocessing.dock import get_reaction_attention_emd
from enum import Enum
from collections import defaultdict
import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import os
from typing import List

n_cpu = os.cpu_count()
IGNORE_DUPLICATES = 0
SAVE_DUPLICATES = 1
DROP_DUPLICATES = 2


def get_ec_map(split):
    with open(f"datasets/ecreact/level4/src-{split}.txt") as f:
        src_lines = f.read().splitlines()
        src_ec = [x.split("|") for x in src_lines]
        src_lines = [x[0] for x in src_ec]
        ec_lines = [x[1] for x in src_ec]
    with open(f"datasets/ecreact/level4/tgt-{split}.txt") as f:
        tgt_lines = f.read().splitlines()
    assert len(src_lines) == len(tgt_lines) == len(ec_lines)
    mapping = {(src, tgt): ec for src, tgt, ec in zip(src_lines, tgt_lines, ec_lines)}
    return mapping


class ECType(Enum):
    NO_EC = 0
    PAPER = 1
    PRETRAINED = 2
    DAE = 3


def get_ec_type_from_num(num):
    return ECType(num)


DEFAULT_EMB_VALUE = torch.tensor([0.0] * 2560).float()


def get_ec_type(use_ec, ec_split, dae):
    if dae:
        return ECType.DAE
    if not use_ec:
        return ECType.NO_EC
    if ec_split:
        return ECType.PAPER
    return ECType.PRETRAINED


class DuplicateSrcManager:

    def __init__(self, base_dataset="datasets/ecreact/level4", threshold=1):
        self.base_dataset = base_dataset
        self.threshold = threshold
        self.duplicates = self.get_duplicates()

    def remove_ec(self, text):
        return text.split("|")[0]

    def get_duplicates(self):
        srcs_counter = Counter()
        for split in ["train", "valid", "test"]:
            with open(f"{self.base_dataset}/src-{split}.txt") as f:
                src_lines = f.read().splitlines()
                src = [self.remove_ec(x) for x in src_lines]
                srcs_counter.update(src)
        print(len(srcs_counter))
        return srcs_counter

    def is_duplicate(self, src):
        return self.duplicates[self.remove_ec(src)] > self.threshold

    def mask_list_duplicate(self, src_list):
        return [self.is_duplicate(src) for src in src_list]


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, weights=None, max_length=200, DEBUG=False,
                 ec_type=ECType.NO_EC, sample_size=None, shuffle=True, alpha=0.5, addec=False, save_ec=False,
                 retro=False, duplicated_source_mode=IGNORE_DUPLICATES):
        self.max_length = max_length
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.retro = retro
        self.addec = addec
        self.data = []
        self.DEBUG = DEBUG
        self.ec_type = ec_type
        self.alpha = alpha
        self.duplicated_source_manager = DuplicateSrcManager()
        if save_ec:
            self.ec_map = get_ec_map(split)
            self.all_ecs = []
        else:
            self.ec_map = None
            self.all_ecs = []

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
            dups = duplicated_source_mode
            have_ec = "ec" in ds
            if "quant" in ds:
                have_ec = False  # TODO : there is EC , but like paper, not like pretrained
            if "uspto" in ds:
                dups = IGNORE_DUPLICATES
            self.load_dataset(f"datasets/{ds}", split, w, have_ec=have_ec,
                              duplicated_source_mode=dups)
        if not DEBUG:
            # if sample_size is not None:
            #     self.data = random.sample(self.data, sample_size)
            if shuffle:
                random.seed(42)
                random.shuffle(self.data)

    def process_reaction(self, text, ec):
        return get_reaction_attention_emd(text, ec, self.ec_to_uniprot, self.smiles_to_id, alpha=self.alpha)

    def load_dataset(self, input_base, split, w, have_ec=True, duplicated_source_mode=0):
        if not os.path.exists(input_base):
            print(f"Dataset {input_base} not found")
            input_base = input_base.replace("-0.5", "")
        with open(f"{input_base}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()

        with open(f"{input_base}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()

        emb_lines = [DEFAULT_EMB_VALUE] * len(src_lines)

        if duplicated_source_mode != IGNORE_DUPLICATES:
            print("Removing duplicates mode ", duplicated_source_mode, ". len before", len(src_lines), end=" ")
            mask = self.duplicated_source_manager.mask_list_duplicate(src_lines)
            if duplicated_source_mode == DROP_DUPLICATES:
                mask = [not x for x in mask]
            src_lines = [src for src, m in zip(src_lines, mask) if m]
            tgt_lines = [tgt for tgt, m in zip(tgt_lines, mask) if m]
            emb_lines = [emb for emb, m in zip(emb_lines, mask) if m]
            print("len after", len(src_lines))

        if self.retro:
            src_lines, tgt_lines = tgt_lines, src_lines
        assert len(src_lines) == len(tgt_lines)

        if self.sample_size is not None:
            samples_idx = random.sample(range(len(src_lines)), self.sample_size)
            src_lines = [src_lines[i] for i in samples_idx]
            tgt_lines = [tgt_lines[i] for i in samples_idx]
            emb_lines = [emb_lines[i] for i in samples_idx]

        if self.ec_map is not None:
            save_ec_lines = [self.ec_map[(src.split("|")[0], tgt)] for src, tgt in zip(src_lines, tgt_lines)]
        else:
            save_ec_lines = [0] * len(src_lines)

        if have_ec:
            if self.ec_type == ECType.NO_EC:
                src_lines = [remove_ec(text) for text in src_lines]
                tgt_lines = [remove_ec(text) for text in tgt_lines]
            elif self.ec_type == ECType.PRETRAINED or self.ec_type == ECType.DAE:
                if self.addec:
                    first_src_lines = src_lines[:]
                src_ec = [redo_ec_split(text, True) for text in src_lines]
                src_lines = [x[0] for x in src_ec]
                ec_lines = [x[1] for x in src_ec]
                if self.ec_type == ECType.PRETRAINED:
                    emb_lines = [self.ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]
                else:
                    # emb_lines = [
                    #     get_reaction_attention_emd(text, ec, self.ec_to_uniprot, self.smiles_to_id, alpha=self.alpha)
                    #     for text, ec in tqdm(zip(src_lines, ec_lines), total=len(src_lines))
                    # ]
                    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
                        emb_lines = list(
                            tqdm(executor.map(self.process_reaction, src_lines, ec_lines), total=len(src_lines)))

                if self.addec:
                    src_lines = first_src_lines
                not_none_mask = [x is not None for x in emb_lines]
                len_before = len(src_lines)
                src_lines = [src_lines[i] for i in range(len(src_lines)) if not_none_mask[i]]
                tgt_lines = [tgt_lines[i] for i in range(len(tgt_lines)) if not_none_mask[i]]
                emb_lines = [emb_lines[i] for i in range(len(emb_lines)) if not_none_mask[i]]
                save_ec_lines = [save_ec_lines[i] for i in range(len(save_ec_lines)) if not_none_mask[i]]
                len_after = len(src_lines)
                print(f"Removed {len_before - len_after} samples, total: {len_after}, {len_before}")

        if self.DEBUG:
            src_lines = src_lines[:100]
            tgt_lines = tgt_lines[:100]
            emb_lines = emb_lines[:100]
            save_ec_lines = save_ec_lines[:100]
        assert len(src_lines) == len(tgt_lines) == len(emb_lines) == len(save_ec_lines)
        skip_count = 0
        data = []
        ec_final = []
        for i in tqdm(range(len(src_lines))):
            input_id = encode_eos_pad(self.tokenizer, src_lines[i], self.max_length, no_pad=True)
            label = encode_eos_pad(self.tokenizer, tgt_lines[i], self.max_length, no_pad=True)
            if input_id is None or label is None:
                skip_count += 1
                continue
            # label[label_mask == 0] = -100

            emb = emb_lines[i]
            if isinstance(emb, torch.Tensor):
                emb = emb.float()
            else:  # numpy array
                emb = torch.tensor(emb).float()
            data.append((input_id, label, emb))
            ec_final.append(save_ec_lines[i])
        for _ in range(w):
            self.data.extend(data)
            self.all_ecs.extend(ec_final)

    def __len__(self):
        return len(self.data)

    def data_to_dict(self, data):
        return {"input_ids": data[0], "labels": data[1], "emb": data[2]}

    def __getitem__(self, idx):
        return self.data_to_dict(self.data[idx])


def combine_datasets(datasets: List[SeqToSeqDataset], shuffle=True) -> SeqToSeqDataset:
    combined_data = []
    combined_ecs = []
    for dataset in datasets:
        combined_data.extend(dataset.data)
        combined_ecs.extend(dataset.all_ecs)

    combined_dataset = SeqToSeqDataset(
        datasets=[],
        split=None,
        tokenizer=None,
        max_length=None,
        DEBUG=None,
        ec_type=None,
        sample_size=None,
        shuffle=shuffle,
        alpha=None,
        addec=None,
        save_ec=None,
        retro=None,
        duplicated_source_mode=None
    )
    if shuffle:
        random.seed(42)
        indexes = list(range(len(combined_data)))
        random.shuffle(indexes)
        combined_data = [combined_data[i] for i in indexes]
        combined_ecs = [combined_ecs[i] for i in indexes]
    combined_dataset.data = combined_data
    combined_dataset.all_ecs = combined_ecs
    return combined_dataset
