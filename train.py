import torch
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import PreTrainedTokenizerFast

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from rdkit import Chem


def disable_rdkit_logging() -> None:
    """Disables RDKit whiny logging."""
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl

    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")


disable_rdkit_logging()

def encode_bos_eos_pad(tokenizer, text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) > max_length - 2:
        return None, None
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    n_tokens = len(tokens)
    padding_length = max_length - len(tokens)
    if padding_length > 0:
        tokens = tokens + [tokenizer.pad_token_id] * padding_length
    mask = [1] * n_tokens + [0] * padding_length
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)
    return tokens, mask


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, weights=None, max_length=200):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = []
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
        if DEBUG:
            src_lines = src_lines[:10]
            tgt_lines = tgt_lines[:10]
        skip_count = 0
        data = []
        for i in tqdm(range(len(src_lines))):
            input_id, attention_mask = encode_bos_eos_pad(self.tokenizer, src_lines[i], self.max_length)
            label, label_mask = encode_bos_eos_pad(self.tokenizer, tgt_lines[i], self.max_length)
            if input_id is None or label is None:
                skip_count += 1
                continue
            label[label_mask == 0] = -100
            data.append((input_id, attention_mask, label, label_mask))
        for _ in range(w):
            self.data.extend(data)

    def __len__(self):
        return len(self.data)

    def data_to_dict(self, data):
        return {"input_ids": data[0], "attention_mask": data[1], "labels": data[2], "decoder_attention_mask": data[3]}

    def __getitem__(self, idx):
        return self.data_to_dict(self.data[idx])


def compute_metrics(eval_pred, tokenizer):
    predictions_, labels_ = eval_pred
    predictions_ = np.argmax(predictions_[0], axis=-1)
    token_acc = []
    accuracy = []
    is_valid = []
    for i in range(len(predictions_)):
        mask = (labels_[i] != tokenizer.pad_token_id) & (labels_[i] != -100)
        pred = predictions_[i][mask]
        label = labels_[i][mask]
        token_acc.append((pred == label).mean().item())
        pred = tokenizer.decode(pred, skip_special_tokens=True)
        is_valid.append(Chem.MolFromSmiles(pred.replace(" ", "")) is not None)
        label = tokenizer.decode(label, skip_special_tokens=True)
        accuracy.append(pred == label)

    token_acc = np.mean(token_acc)
    accuracy = np.mean(accuracy)
    is_valid = np.mean(is_valid)
    return {"accuracy": accuracy, "valid_smiles": is_valid, "token_acc": token_acc}


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("datasets/tokenizer")
    n_add = tokenizer.add_special_tokens({"bos_token": "<BOS>", "eos_token": "<EOS>", "pad_token": "<PAD>"})

    config = T5Config(vocab_size=n_add + len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                      decoder_start_token_id=tokenizer.bos_token_id)
    if DEBUG:
        config.num_layers = 1
        config.d_model = 128
        config.num_heads = 4
        config.d_ff = 256
    model = T5ForConditionalGeneration(config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_dataset = SeqToSeqDataset(["uspto", 'ecreact'], "train", weights=[1, 9], tokenizer=tokenizer)
    eval_split = "valid" if not DEBUG else "train"
    val_ecreact = SeqToSeqDataset(["ecreact"], eval_split, weights=[1], tokenizer=tokenizer)
    val_uspto = SeqToSeqDataset(["uspto"], eval_split, weights=[1], tokenizer=tokenizer)
    eval_datasets = {"ecreact": val_ecreact, "uspto": val_uspto}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="results",
        evaluation_strategy="steps",
        save_steps=5_000,
        save_total_limit=20,
        max_steps=500_000,
        auto_find_batch_size=True,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=64,
        logging_steps=500 if not DEBUG else 10,
        eval_steps=500 if not DEBUG else 10,
        warmup_steps=8_000 if not DEBUG else 10,
        eval_accumulation_steps=8,
        report_to='none' if DEBUG else 'tensorboard',
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG = args.debug
    main()
