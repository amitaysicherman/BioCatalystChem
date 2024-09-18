from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from preprocessing.build_tokenizer import redo_ec_split, encode_bos_eos_pad
from utils import remove_ec


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, weights=None, max_length=200, use_ec=True,
                 ec_split=True, DEBUG=False):
        self.use_ec = use_ec
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ec_split = ec_split
        self.data = []
        self.DEBUG = DEBUG

        if weights is None:
            weights = [1] * len(datasets)
        else:
            assert len(weights) == len(datasets)
        for ds, w in zip(datasets, weights):
            self.load_dataset(f"datasets/{ds}", split, w)

    def load_dataset(self, input_base, split, w):
        with open(f"{input_base}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        with open(f"{input_base}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        assert len(src_lines) == len(tgt_lines)
        if not self.use_ec:
            src_lines = [remove_ec(text) for text in src_lines]
            tgt_lines = [remove_ec(text) for text in tgt_lines]
        if not self.ec_split:
            src_lines = [redo_ec_split(text) for text in src_lines]
            tgt_lines = [redo_ec_split(text) for text in tgt_lines]
        if self.DEBUG:
            src_lines = src_lines[:1]
            tgt_lines = tgt_lines[:1]
        skip_count = 0
        data = []
        for i in tqdm(range(len(src_lines))):
            input_id, attention_mask = encode_bos_eos_pad(self.tokenizer, src_lines[i], self.max_length)
            label, label_mask = encode_bos_eos_pad(self.tokenizer, tgt_lines[i], self.max_length)
            if input_id is None or label is None:
                skip_count += 1
                continue
            label[label_mask == 0] = -100
            data.append((input_id, attention_mask, label))
        for _ in range(w):
            self.data.extend(data)

    def __len__(self):
        return len(self.data)

    def data_to_dict(self, data):
        return {"input_ids": data[0], "attention_mask": data[1], "labels": data[2]}

    def __getitem__(self, idx):
        return self.data_to_dict(self.data[idx])
