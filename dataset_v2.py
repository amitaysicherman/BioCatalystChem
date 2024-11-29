import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from preprocessing.build_tokenizer import get_ec_from_seq, encode_eos_pad
from utils import remove_ec
from preprocessing.dock import Docker

from enum import Enum
from collections import defaultdict
import pandas as pd
import random
from tqdm import tqdm
from collections import Counter
import os
from typing import List


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, add_emb, weights=None, max_length=200,
                 DEBUG=False,sample_size=None, shuffle=True):
        self.max_length = max_length
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.data = []
        self.DEBUG = DEBUG
        self.samples_ids = []

        self.ec_to_vec = Docker()
        # with open("datasets/docking/smiles_to_id.txt") as f:
        #     self.smiles_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
        ec_mapping = pd.read_csv("datasets/ec_map.csv")
        self.ec_to_uniprot = defaultdict(str)
        for i, row in ec_mapping.iterrows():
            self.ec_to_uniprot[row["EC_full"]] = row["Uniprot_id"]

        if weights is None:
            weights = [1] * len(datasets)
        else:
            assert len(weights) == len(datasets)
        assert len(datasets) == len(add_emb)
        for ds, w, ae in zip(datasets, weights, add_emb):
            dataset = self.load_dataset(f"datasets/{ds}", split, ae)
            for _ in range(w):
                self.data.extend(dataset)
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        print(f"Dataset {split} loaded, len: {len(self.data)}")

    def load_dataset(self, input_base, split, add_emb):
        with open(f"{input_base}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()

        with open(f"{input_base}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()

        if self.sample_size is not None:
            src_lines, tgt_lines = zip(*random.sample(list(zip(src_lines, tgt_lines)), self.sample_size))

        ec_lines = [get_ec_from_seq(text, True) for text in src_lines]
        if add_emb:
            uniprot_ids = [self.ec_to_uniprot[ec] if ec in self.ec_to_uniprot else None for ec in ec_lines]
            files_pathed = [f"datasets/docking/{uniprot_id}/protein.npy" for uniprot_id in uniprot_ids]
            emb_lines = [np.load(f) if os.path.exists(f) else None for f in files_pathed]

        if self.ec_type == ECType.PRETRAINED:
            emb_lines = [self.ec_to_vec.ec_to_vec_mem.get(ec, None) for ec in tqdm(ec_lines)]
        else:
            emb_lines = [
                # get_reaction_attention_emd(text, ec, self.ec_to_uniprot, self.smiles_to_id, alpha=self.alpha,
                #                            v2=self.daev2)
                # for text, ec in tqdm(zip(src_lines, ec_lines), total=len(src_lines))
                self.ec_to_vec.dock_src_ec(text, ec) for text, ec in
                tqdm(zip(src_lines, ec_lines), total=len(src_lines))
            ]
            # with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            #     emb_lines = list(
            #         tqdm(executor.map(self.process_reaction, src_lines, ec_lines), total=len(src_lines)))

        if self.addec:
            src_lines = first_src_lines
        not_none_mask = [x is not None for x in emb_lines]
        len_before = len(src_lines)
        src_lines = [src_lines[i] for i in range(len(src_lines)) if not_none_mask[i]]
        tgt_lines = [tgt_lines[i] for i in range(len(tgt_lines)) if not_none_mask[i]]
        emb_lines = [emb_lines[i] for i in range(len(emb_lines)) if not_none_mask[i]]
        save_ec_lines = [save_ec_lines[i] for i in range(len(save_ec_lines)) if not_none_mask[i]]
        source_lines = [source_lines[i] for i in range(len(source_lines)) if not_none_mask[i]]
        sid = [sid[i] for i in range(len(sid)) if not_none_mask[i]]
        len_after = len(src_lines)
        print(f"Removed {len_before - len_after} samples, total: {len_after}, {len_before}")

        if self.DEBUG:
            src_lines = src_lines[:100]
            tgt_lines = tgt_lines[:100]
            emb_lines = emb_lines[:100]
            save_ec_lines = save_ec_lines[:100]
            source_lines = source_lines[:100]
            sid = sid[:100]
        assert len(src_lines) == len(tgt_lines) == len(emb_lines) == len(save_ec_lines) == len(source_lines) == len(sid)
        skip_count = 0
        data = []
        ec_final = []
        source_final = []
        sid_final = []
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
            source_final.append(source_lines[i])
            sid_final.append(sid[i])

        for _ in range(w):
            self.data.extend(data)
            self.all_ecs.extend(ec_final)
            self.sources.extend(source_final)
            self.samples_ids.extend(sid_final)

    def __len__(self):
        return len(self.data)

    def data_to_dict(self, data):
        return {"input_ids": data[0], "labels": data[1], "emb": data[2]}

    def __getitem__(self, idx):
        return self.data_to_dict(self.data[idx])


def combine_datasets(datasets: List[SeqToSeqDataset], shuffle=True) -> SeqToSeqDataset:
    combined_data = []
    combined_ecs = []
    combined_sources = []
    for dataset in datasets:
        combined_data.extend(dataset.data)
        combined_ecs.extend(dataset.all_ecs)
        combined_sources.extend(dataset.sources)

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
        combined_sources = [combined_sources[i] for i in indexes]

    combined_dataset.data = combined_data
    combined_dataset.all_ecs = combined_ecs
    combined_dataset.sources = combined_sources
    return combined_dataset


if "__main__" == __name__:
    class tok:
        def __init__(self):
            self.eos_token_id = 12
            self.pad_token_id = 0
            self.eos_token_id = 12

        def encode(self, s, **kwargs):
            return [ord(c) for c in s]

        def decode(self, s, **kwargs):
            return "".join([chr(c) for c in s])


    t = tok()
    ds = SeqToSeqDataset(datasets=["ecreact/level4"], split="test", tokenizer=t, ec_type=ECType.PRETRAINED,
                         ec_source="all", save_ec=True, max_length=500, drop_short=True)
    print(ds.samples_ids)
