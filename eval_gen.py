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
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def eval_gen(model:T5ForConditionalGeneration, tokenizer, dataloader, output_file, top_k=[1, 3, 5, 10]):
    gt_list = []
    preds_list = []

    max_k = max(top_k)
    pbar = tqdm(dataloader)
    top_k_correct = defaultdict(int)
    for batch in pbar:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=tokenizer.model_max_length, do_sample=False, num_beams=max_k * 2,
                                 num_return_sequences=max_k)
        labels = [l[l != -100] for l in batch['labels'].cpu().numpy()]

        for i in range(len(labels)):
            gt_list.append(tokens_to_canonical_smiles(tokenizer, labels[i]))
            preds_list.append([tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs[i]])
            for k in top_k:
                if gt_list[-1] in preds_list[-1][:k]:
                    top_k_correct[k] += 1
            text = "\n".join([f"GT: {gt_list[-1]}, PR: {pred}" for pred in preds_list[-1]]) + "\n"
            if output_file != "":
                with open(output_file, "a") as f:
                    f.write(text)
            else:
                print(text)

            top_k_acc = {k: top_k_correct[k] / len(gt_list) for k in top_k}
            pbar.set_description(f"Top-k Acc: {top_k_acc}")

    return {k: top_k_correct[k] / len(gt_list) for k in top_k}


def prep_gen_dirs(summary_file):
    if not os.path.exists("gen"):
        os.makedirs("gen")
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write("dataset,cp,acc,k\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="paper", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", default="uspto", type=str)
    args = parser.parse_args()

    cp_dir_all = f"results/{args.run_name}"
    cp_dir = sorted([f for f in os.listdir(cp_dir_all) if re.match(r"checkpoint-\d+", f)],
                    key=lambda x: int(x.split("-")[1]))[-1]
    cp_dir = f"{cp_dir_all}/{cp_dir}"
    run_args = name_to_args(args.run_name)

    ec_split = run_args["ec_split"]
    lookup_len = run_args["lookup_len"]

    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path(ec_split))
    config = T5Config.from_pretrained(cp_dir + "/config.json")

    if ec_split:
        model = T5ForConditionalGeneration(config)
    else:
        ec_order = get_ec_order(tokenizer, ec_split)
        cutoff_index = get_first_ec_token_index(tokenizer, ec_split)
        config.vocab_size = cutoff_index
        model = CustomT5Model(config, lookup_len, cutoff_index, ec_order)
    model.load_state_dict(torch.load(f"{cp_dir}/pytorch_model.bin", map_location="cpu"))
    model.to(device)
    model.eval()

    if args.debug:
        sample_size = 100
    else:
        sample_size = None
    gen_dataset = SeqToSeqDataset([args.dataset], "valid", tokenizer=tokenizer, use_ec=run_args["use_ec"],
                                  ec_split=ec_split, DEBUG=args.debug)
    gen_dataloader = DataLoader(gen_dataset, batch_size=8, num_workers=0)
    output_file = f"gen/{args.dataset}${args.run_name}.txt" if not args.debug else ""
    if output_file and os.path.exists(output_file):
        os.remove(output_file)
    summary_file = f"gen/summary.csv"
    prep_gen_dirs(summary_file)
    with torch.no_grad():
        results_dict = eval_gen(model, tokenizer, gen_dataloader, output_file)
    if args.debug_mode:
        print("-----------------")
        for k, acc in results_dict.items():
            print(f"{k}: {acc}")
        print("-----------------")
    else:
        with open(summary_file, "a") as f:
            for k, acc in results_dict.items():
                f.write(f"{args.dataset},{args.run_name},{acc},{k}\n")
