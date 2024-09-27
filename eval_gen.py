from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration, T5Config
import numpy as np
import argparse
from rdkit import Chem
from collections import defaultdict
from torch.utils.data import DataLoader
from dataset import SeqToSeqDataset
from preprocessing.build_tokenizer import get_tokenizer_file_path, get_first_ec_token_index, get_ec_order
from model import CustomT5Model
import torch
import os
import re
from tqdm import tqdm
from rdkit import RDLogger
from train import name_to_args

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokens_to_canonical_smiles(tokenizer, tokens):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def eval_dataset(model:T5ForConditionalGeneration,gen_dataloader:DataLoader,k=10):
    correct_count= {i:0 for i in range(1,k+1)}
    pbar=tqdm(enumerate(gen_dataloader),total=len(gen_dataloader))
    for i, batch in pbar:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device).bool()
        labels = batch['labels'].to(model.device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=200, do_sample=False, num_beams=k * 2,
                                 num_return_sequences=k)
        mask = (labels != tokenizer.pad_token_id) & (labels != -100)
        labels = labels[mask]

        labels=tokens_to_canonical_smiles(tokenizer, labels)
        preds_list=[tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs]
        for j in range(1,k+1):
            if labels in preds_list[:j]:
                correct_count[j] += 1
        msg=" | ".join([f"{j}:{correct_count[j]/(i+1):.2f}" for j in range(1,k+1)])
        msg=f'{i+1}/{len(gen_dataloader)} | {msg}'
        pbar.set_description(msg)
    return {i: correct_count[i] / len(gen_dataloader) for i in [1,3,5,10]}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="paper", type=str)
    args = parser.parse_args()
    dataset = "ecreact/level4"
    run_name = args.run_name
    cp_dir_all = f"results/{run_name}"

    if "pretrained" in run_name:
        ec_split = False
        use_ec = True
        lookup_len = int(run_name.split("_")[1])
        tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path(ec_split))
        model_type = CustomT5Model
        models_args = {"lookup_len": lookup_len, "ec_tokens_order": get_ec_order(tokenizer, ec_split),
                       "cutoff_index": get_first_ec_token_index(tokenizer, ec_split)}
    else:
        if run_name == "regular":
            ec_split = True
            use_ec = False
        elif run_name == "paper":
            ec_split = True
            use_ec = True
        else:
            raise ValueError(f"Unknown run_name: {run_name}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path(ec_split))
        model_type = T5ForConditionalGeneration
        models_args = {}

    gen_dataset = SeqToSeqDataset([dataset], "valid", tokenizer=tokenizer, use_ec=use_ec,
                                  ec_split=ec_split, DEBUG=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=1, num_workers=0)

    cp_dir = sorted([f for f in os.listdir(cp_dir_all) if re.match(r"checkpoint-\d+", f)],
                    key=lambda x: int(x.split("-")[1]))
    cp_dir = cp_dir[-1]
    cp_dir = f"{cp_dir_all}/{cp_dir}"
    model = model_type.from_pretrained(cp_dir, **models_args)
    model.to(device)
    model.eval()
    correct_count = eval_dataset(model, gen_dataloader)
    print(f"Run: {run_name}")
    for k, acc in correct_count.items():
        print(f"{k}: {acc}")
    output_file = f"{cp_dir_all}/eval_{run_name}.txt"
    with open(output_file, "w") as f:
        for k, acc in correct_count.items():
            f.write(f"{k}: {acc}\n")
