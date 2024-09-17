# sbatch --gres=gpu:1 --mem=16G --time=1-00:00:00 --account=def-bengioy --cpus-per-task=4 train.sh
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)
from transformers import Trainer, TrainingArguments

from transformers import PreTrainedTokenizerFast

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from preprocessing.build_tokenizer import get_tokenizer_file_path, get_first_ec_token_index, get_ec_order, \
    redo_ec_split, encode_bos_eos_pad
from ec_prot_model import CustomT5Model


def disable_rdkit_logging() -> None:
    """Disables RDKit whiny logging."""
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl

    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")


disable_rdkit_logging()


def remove_ec(text):
    return text.split("|")[0]


class SeqToSeqDataset(Dataset):
    def __init__(self, datasets, split, tokenizer: PreTrainedTokenizerFast, weights=None, max_length=200, use_ec=True,
                 ec_split=True):
        self.use_ec = use_ec
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ec_split = ec_split
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
        if not self.use_ec:
            src_lines = [remove_ec(text) for text in src_lines]
            tgt_lines = [remove_ec(text) for text in tgt_lines]
        if not self.ec_split:
            src_lines = [redo_ec_split(text) for text in src_lines]
            tgt_lines = [redo_ec_split(text) for text in tgt_lines]
        if DEBUG:
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


def get_last_cp(base_dir):
    import os
    import re
    if not os.path.exists(base_dir):
        return None
    cp_dirs = os.listdir(base_dir)
    cp_dirs = [f for f in cp_dirs if re.match(r"checkpoint-\d+", f)]
    cp_dirs = sorted(cp_dirs, key=lambda x: int(x.split("-")[1]))
    if len(cp_dirs) == 0:
        return None
    return f"{base_dir}/{cp_dirs[-1]}"


def main(use_ec=True, ec_split=False):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path(ec_split))
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                      decoder_start_token_id=tokenizer.bos_token_id)
    if DEBUG:
        config.num_layers = 1
        config.d_model = 128
        config.num_heads = 4
        config.d_ff = 256
    if ec_split:
        model = T5ForConditionalGeneration(config)
    else:
        ec_order = get_ec_order(tokenizer, ec_split)
        cutoff_index = get_first_ec_token_index(tokenizer, ec_split)
        lookup_len = 5
        config.vocab_size = cutoff_index
        model = CustomT5Model(config, lookup_len, cutoff_index, ec_order)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    ecreact_dataset = "ecreact/level3" if ec_split else "ecreact/level4"
    train_dataset = SeqToSeqDataset(["uspto", ecreact_dataset], "train", weights=[1, 9], tokenizer=tokenizer,
                                    use_ec=use_ec,
                                    ec_split=ec_split)
    eval_split = "valid" if not DEBUG else "train"

    val_ecreact = SeqToSeqDataset([ecreact_dataset], eval_split, weights=[1], tokenizer=tokenizer, use_ec=use_ec,
                                  ec_split=ec_split)
    val_uspto = SeqToSeqDataset(["uspto"], eval_split, weights=[1], tokenizer=tokenizer, use_ec=use_ec,
                                ec_split=ec_split)
    eval_datasets = {"ecreact": val_ecreact, "uspto": val_uspto}
    if use_ec:
        if ec_split:
            run_name = "paper"
        else:
            run_name = "pretrained"
    else:
        run_name = "regular"
    print(f"Run name: {run_name}")
    # Training arguments
    output_dir = f"results/{run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_steps=5_000 if not DEBUG else 10,
        save_total_limit=10,
        max_steps=500_000,
        auto_find_batch_size=True,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024 // 8,
        logging_steps=500 if not DEBUG else 10,
        eval_steps=25_000 if not DEBUG else 10,
        warmup_steps=8_000 if not DEBUG else 10,
        eval_accumulation_steps=8,
        report_to='none' if DEBUG else 'tensorboard',
        run_name=run_name,
        resume_from_checkpoint=get_last_cp(output_dir)
    )

    # Initialize Trainer
    trainer = Trainer(
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
    parser.add_argument("--use_ec", default=1, type=int)
    parser.add_argument("--ec_split", default=0, type=int)
    args = parser.parse_args()
    DEBUG = args.debug
    main(args.use_ec, args.ec_split)
