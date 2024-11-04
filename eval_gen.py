import pandas as pd
from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration
import argparse
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset import SeqToSeqDataset, get_ec_type
from preprocessing.build_tokenizer import get_tokenizer_file_path, get_first_ec_token_index, get_ec_order
from model import CustomT5Model
import torch
import os
import re
from tqdm import tqdm
from rdkit import RDLogger
import json
from dataset import ECType
from preprocessing.build_tokenizer import get_ec_tokens

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from enum import Enum


def name_to_args(name):
    # Initialize default values for the arguments
    args = {
        "ec_type": None,
        "lookup_len": None,
        "prequantization": False,
        "n_hierarchical_clusters": None,
        "n_pca_components": None,
        "n_clusters_pca": None,
        "alpha": None,
        'addec': False
    }
    if "plus" in name:
        args["addec"] = True

    name = name.replace("_mix", "").replace("_nopre", "").replace("_regpre", "").replace("_plus", "")

    if name == "uspto":
        args["ec_type"] = ECType.NO_EC
        return args

    if name == "paper":
        args["ec_type"] = ECType.PAPER
        return args
    elif name == "regular":
        args["ec_type"] = ECType.NO_EC
        return args
    elif name.startswith("pretrained"):
        args["ec_type"] = ECType.PRETRAINED
    elif name.startswith("dae"):
        args["ec_type"] = ECType.DAE
        ec_alpha = name.split("_")[0]
        if "-" in ec_alpha:
            args["alpha"] = float(ec_alpha.split("-")[1])
        else:
            args["alpha"] = 0.5
    # Check if the name contains "quant" (prequantization is True)
    if "_quant" in name:
        args["prequantization"] = True
        # Extract hierarchical clusters, PCA components, and clusters from the name
        parts = name.split("_quant_")[1].split("_")
        args["n_hierarchical_clusters"] = int(parts[0])
        args["n_pca_components"] = int(parts[1])
        args["n_clusters_pca"] = int(parts[2])
    else:
        # Extract lookup_len from the name if no quantization is used
        args["lookup_len"] = int(name.split("_")[-1])

    return args


def tokens_to_canonical_smiles(tokenizer, tokens):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def eval_dataset(model: T5ForConditionalGeneration, gen_dataloader: DataLoader, k=10, fast=0, save_file=None,
                 all_ec=None):
    correct_count = {ec: {i: 0 for i in range(1, k + 1)} for ec in set(all_ec)}
    ec_count = {ec: 0 for ec in set(all_ec)}
    pbar = tqdm(enumerate(gen_dataloader), total=len(gen_dataloader))
    for i, batch in pbar:
        ec = all_ec[i]
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device).bool()
        labels = batch['labels'].to(model.device)
        emb = batch['emb'].to(model.device).float()
        res = 0
        if (emb == 0).all():
            emb_args = {}
        else:
            emb_args = {"emb": emb}
        if fast:  # predicnt and not generate
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **emb_args)
            predictions = outputs.logits.argmax(dim=-1)
            mask = (labels != tokenizer.pad_token_id) & (labels != -100)
            pred = predictions[mask]
            label = labels[mask]
            pred = tokenizer.decode(pred, skip_special_tokens=True)
            label = tokenizer.decode(label, skip_special_tokens=True)
            correct_count[ec][1] += (pred == label)
            ec_count[ec] += 1
            y = tokenizer.decode(labels[labels != -100], skip_special_tokens=True)
            x = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            with open(save_file, "a") as f:
                f.write(f"{x},{y},{label == pred}\n")

        else:
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_length=200, do_sample=False, num_beams=k,
                                     num_return_sequences=k, **emb_args)
            mask = (labels != tokenizer.pad_token_id) & (labels != -100)
            labels = labels[mask]

            labels = tokens_to_canonical_smiles(tokenizer, labels)
            preds_list = [tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs]
            for j in range(1, k + 1):
                if labels in preds_list[:j]:
                    correct_count[ec][j] += 1
            ec_count[ec] += 1
        msg = ""
        for ec in correct_count:
            msg += f"[{ec}: "
            for i in range(1, k + 1):
                if ec_count[ec] == 0:
                    continue
                msg += f"{i}:{correct_count[ec][i] / ec_count[ec]:.2f} "
            msg += f"|{ec_count[ec]:.2f}] "
        pbar.set_description(msg)
    return correct_count, ec_count


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


def get_best_val_cp(run_name):
    base_dir = f"results/{run_name}"
    last_cp = get_last_cp(base_dir)
    trainer_state_file = f"{last_cp}/trainer_state.json"
    if not os.path.exists(trainer_state_file):
        raise ValueError(f"trainer_state.json not found in {base_dir}")
    with open(trainer_state_file) as f:
        trainer_state = json.load(f)
    return trainer_state["best_model_checkpoint"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="pretrained_5", type=str)
    parser.add_argument("--split", default="valid", type=str)
    parser.add_argument("--fast", default=1, type=int)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--per_level", default=1, type=int)

    args = parser.parse_args()
    run_name = args.run_name
    run_args = name_to_args(run_name)
    print("---" * 10)
    print(f"Run: {run_name}")
    print(run_args)
    print("---" * 10)
    ec_type = run_args["ec_type"]
    lookup_len = run_args["lookup_len"]
    prequantization = run_args["prequantization"]
    n_hierarchical_clusters = run_args["n_hierarchical_clusters"]
    n_pca_components = run_args["n_pca_components"]
    n_clusters_pca = run_args["n_clusters_pca"]
    addec = run_args["addec"]
    alpha = run_args["alpha"]
    per_level = args.per_level
    if prequantization:
        from offline_quantizer import args_to_quant_dataset

        ecreact_dataset = args_to_quant_dataset(ec_type, n_hierarchical_clusters,
                                                n_pca_components, n_clusters_pca, alpha)
        ecreact_dataset = ecreact_dataset.replace("datasets/", "")
    else:
        ecreact_dataset = "ecreact/level4"
    if addec:
        ecreact_dataset += "_plus"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())

    if prequantization:
        from offline_quantizer import HierarchicalPCATokenizer

        new_tokens = HierarchicalPCATokenizer(n_hierarchical_clusters=n_hierarchical_clusters,
                                              n_pca_components=n_pca_components,
                                              n_clusters_pca=n_clusters_pca,
                                              ).get_all_tokens()
        tokenizer.add_tokens(new_tokens)
    elif ec_type == ECType.PAPER or addec:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    best_val_cp = get_best_val_cp(run_name)

    if (ec_type == ECType.PAPER or ec_type == ec_type.NO_EC) or prequantization:
        model = T5ForConditionalGeneration.from_pretrained(best_val_cp)
    else:
        print("Loading custom model", best_val_cp)
        model = CustomT5Model.from_pretrained(best_val_cp, lookup_len=lookup_len)
    gen_dataset = SeqToSeqDataset([ecreact_dataset], args.split, tokenizer=tokenizer, ec_type=ec_type, DEBUG=False,
                                  save_ec=True)
    all_ec = gen_dataset.all_ecs
    if per_level != 0:
        all_ec = [" ".join(ec.split(" ")[:per_level]) for ec in all_ec]

    all_ec=all_ec[:100]
    gen_dataset.data=gen_dataset.data[:100]

    gen_dataloader = DataLoader(gen_dataset, batch_size=1, num_workers=0)

    model.to(device)
    model.eval()

    # Evaluate the averaged model
    os.makedirs("results/full", exist_ok=True)
    correct_count, ec_count = eval_dataset(model, gen_dataloader, k=args.k, fast=args.fast,
                                           save_file=f"results/full/{run_name}.csv", all_ec=all_ec)
    print(f"Run: {run_name}")
    for ec in correct_count:
        for i in range(1, args.k + 1):
            if ec_count[ec] == 0:
                continue
            print(f"{ec}: {i}: {correct_count[ec][i] / ec_count[ec]:.2f}")
        print(f"{ec}: {ec_count[ec]:.2f}")
    # Save the evaluation results
    output_file = f"results/eval_gen.csv"
    with open(output_file, "a") as f:  # Changed to append mode to log multiple runs
        for ec in correct_count:
            for i in range(1, args.k + 1):
                if ec_count[ec] == 0:
                    continue
                f.write(
                    run_name + "," + args.split + "," + ec + f",{i},{correct_count[ec][i] / ec_count[ec]:.2f},{ec_count[ec]}\n")
