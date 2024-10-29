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
        "alpha": None
    }

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
        ec_alpha = name.split("-")[0]
        if "-" in ec_alpha:
            args["alpha"] = int(ec_alpha.split("-")[1]) / 100
        else:
            args["alpha"] = 50 / 100
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


def eval_dataset(model: T5ForConditionalGeneration, gen_dataloader: DataLoader, k=10, fast=0, save_file=None):
    correct_count = {i: 0 for i in range(1, k + 1)}
    pbar = tqdm(enumerate(gen_dataloader), total=len(gen_dataloader))
    for i, batch in pbar:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device).bool()
        labels = batch['labels'].to(model.device)
        emb = batch['emb'].to(model.device).float()
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
            correct_count[1] += (pred == label)
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
                    correct_count[j] += 1
        y = tokenizer.decode(labels[labels!=-100], skip_special_tokens=True)
        x = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        with open(save_file, "a") as f:
            f.write(f"{x},{y}\n")
        msg = " | ".join([f"{j}:{correct_count[j] / (i + 1):.2f}" for j in range(1, k + 1)])
        msg = f'{i + 1}/{len(gen_dataloader)} | {msg}'
        pbar.set_description(msg)
    return {i: correct_count[i] / len(gen_dataloader) for i in [1, 3, 5, 10]}


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
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--fast", default=1, type=int)
    parser.add_argument("--k", default=3, type=int)

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
    alpha = run_args["alpha"]
    if prequantization:
        from offline_quantizer import args_to_quant_dataset

        ecreact_dataset = args_to_quant_dataset(ec_type, n_hierarchical_clusters,
                                                n_pca_components, n_clusters_pca, alpha)
        ecreact_dataset = ecreact_dataset.replace("datasets/", "")
    else:
        ecreact_dataset = "ecreact/level4"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())

    if prequantization:
        from offline_quantizer import HierarchicalPCATokenizer

        new_tokens = HierarchicalPCATokenizer(n_hierarchical_clusters=n_hierarchical_clusters,
                                              n_pca_components=n_pca_components,
                                              n_clusters_pca=n_clusters_pca,
                                              ).get_all_tokens()
        tokenizer.add_tokens(new_tokens)
    elif ec_type == ECType.PAPER:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    best_val_cp = get_best_val_cp(run_name)

    if (ec_type == ECType.PAPER or ec_type == ec_type.NO_EC) or prequantization:
        model = T5ForConditionalGeneration.from_pretrained(best_val_cp)
    else:
        print("Loading custom model", best_val_cp)
        model = CustomT5Model.from_pretrained(best_val_cp, lookup_len=lookup_len)
    gen_dataset = SeqToSeqDataset([ecreact_dataset], args.split, tokenizer=tokenizer, ec_type=ec_type, DEBUG=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=1, num_workers=0)

    model.to(device)
    model.eval()

    # Evaluate the averaged model
    os.makedirs("results/full", exist_ok=True)
    correct_count = eval_dataset(model, gen_dataloader, k=args.k, fast=args.fast,
                                 save_file=f"results/full/{run_name}.csv")
    print(f"Run: {run_name}")
    for k, acc in correct_count.items():
        print(f"{k}: {acc}")

    # Save the evaluation results
    output_file = f"results/eval_gen.csv"
    with open(output_file, "a") as f:  # Changed to append mode to log multiple runs
        f.write(run_name + args.split + "," + best_val_cp + "," + ",".join(
            [str(correct_count[i]) for i in [1, 3, 5, 10]]) + "\n")
