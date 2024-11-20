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
from finetune_ecreact import CustomDataCollatorForSeq2Seq
from collections import Counter

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        'addec': False,
        'batch_size': 64,
        'lr': 1e-4,
        'ec_source': None
    }

    # if ec_source:
    #     run_name += f"_ecs-{ec_source}"

    if "_ecs-" in name:
        args["ec_source"] = name.split("_ecs-")[1]
        name = name.replace(f"_ecs-{args['ec_source']}", "")

    if "plus" in name:
        args["addec"] = True
    if "_bs" in name:
        args["batch_size"] = int(name.split("_bs")[1].split("_")[0])
        name = name.replace(f"_bs{args['batch_size']}", "")
    if "_lr" in name:
        args["lr"] = float(name.split("_lr")[1].split("_")[0])
        name = name.replace(f"_lr{args['lr']}", "")
    name = name.replace("_mix", "").replace("_nopre", "").replace("_regpre", "").replace("_plus", "")
    name = name.replace("_dups3", "").replace("_dups2", "").replace("_dups1", "").replace("_dups0", "")
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


def eval_dataset(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizerFast, gen_dataloader: DataLoader,
                 k=10, fast=0, save_file=None,
                 all_ec=None, return_all=False):
    correct_count = {ec: {i: 0 for i in range(1, k + 1)} for ec in set(all_ec)}
    ec_count = {ec: 0 for ec in set(all_ec)}
    if return_all:
        all_scores = []
    pbar = tqdm(enumerate(gen_dataloader), total=len(gen_dataloader))

    for i, batch in pbar:
        batch_ec = all_ec[i * len(batch['input_ids']):(i + 1) * len(batch['input_ids'])]
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device).bool()
        labels = batch['labels'].to(model.device)
        emb = batch['emb'].to(model.device).float()
        if (emb == 0).all():
            emb_args = {}
        else:
            emb_args = {"emb": emb}

        if fast:  # predict and not generate
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **emb_args)
            predictions = outputs.logits.argmax(dim=-1)
            mask = (labels != tokenizer.pad_token_id) & (labels != -100)

            for j in range(len(batch_ec)):  # Iterate over each example in the batch
                pred = predictions[j][mask[j]]
                label = labels[j][mask[j]]

                pred_decoded = tokenizer.decode(pred, skip_special_tokens=True)
                label_decoded = tokenizer.decode(label, skip_special_tokens=True)

                correct_count[batch_ec[j]][1] += (pred_decoded == label_decoded)
                if return_all:
                    all_scores.append((pred_decoded == label_decoded))
                ec_count[batch_ec[j]] += 1

                if save_file:
                    y = tokenizer.decode(labels[j][labels[j] != -100], skip_special_tokens=True)
                    x = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                    with open(save_file, "a") as f:
                        f.write(f"{x},{y},{label_decoded == pred_decoded}\n")

        else:  # Generation with beam search
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_length=200, do_sample=False, num_beams=k,
                                     num_return_sequences=k, **emb_args)

            for j in range(len(batch_ec)):  # Iterate over each example in the batch
                mask = (labels[j] != tokenizer.pad_token_id) & (labels[j] != -100)
                label = labels[j][mask]

                label_smiles = tokens_to_canonical_smiles(tokenizer, label)
                preds_list = [tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs[j * k:(j + 1) * k]]

                for rank in range(1, k + 1):
                    if label_smiles in preds_list[:rank]:
                        correct_count[batch_ec[j]][rank] += 1
                ec_count[batch_ec[j]] += 1

    if return_all:
        return correct_count, ec_count, all_scores
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


def get_closest_cp(base_dir, cp_step):
    cp_dirs = os.listdir(base_dir)
    cp_dirs = [f for f in cp_dirs if re.match(r"checkpoint-\d+", f)]
    cp_dirs = sorted(cp_dirs, key=lambda x: int(x.split("-")[1]))
    cp_steps = [int(cp.split("-")[1]) for cp in cp_dirs]
    cp_diffs = [abs(cp_step - cp) for cp in cp_steps]
    closest_cp = cp_dirs[cp_diffs.index(min(cp_diffs))]
    return f"{base_dir}/{closest_cp}"


def get_best_val_cp(run_name, base_results_dir="results", cp_step=None):
    if cp_step is not None:
        return get_closest_cp(f"{base_results_dir}/{run_name}", cp_step)
    base_dir = f"{base_results_dir}/{run_name}"
    last_cp = get_last_cp(base_dir)
    trainer_state_file = f"{last_cp}/trainer_state.json"
    if not os.path.exists(trainer_state_file):
        raise ValueError(f"trainer_state.json not found in {base_dir}")
    with open(trainer_state_file) as f:
        trainer_state = json.load(f)
    best_model_checkpoint = trainer_state["best_model_checkpoint"]
    if not best_model_checkpoint.startswith(base_results_dir):
        best_model_checkpoint_split = best_model_checkpoint.split("/")
        best_model_checkpoint = f"{base_results_dir}/" + "/".join(best_model_checkpoint_split[1:])
    return best_model_checkpoint


def args_to_lens(args):
    length = 200
    if args['ec_type'] == ECType.PAPER:
        length += 5
    if args['prequantization']:
        length += args['n_hierarchical_clusters'] + args['n_pca_components'] + 1
    if args['addec']:
        length += 5
    return length


def get_only_new_ecs(eval_dataset: SeqToSeqDataset):
    with open("datasets/ecreact/level4/src-train.txt") as f:
        src_lines = f.read().splitlines()
    train_ec = [x.split("|")[1].strip() for x in src_lines]
    eval_ecs = [x.strip() for x in eval_dataset.all_ecs]
    eval_mask = [x not in train_ec for x in eval_ecs]
    if eval_dataset.sources is not None:
        eval_dataset.sources = [x for i, x in enumerate(eval_dataset.sources) if eval_mask[i]]
    if eval_dataset.all_ecs is not None:
        eval_dataset.all_ecs = [x for i, x in enumerate(eval_dataset.all_ecs) if eval_mask[i]]
    eval_dataset.data = [x for i, x in enumerate(eval_dataset.data) if eval_mask[i]]
    return eval_dataset


def load_model_tokenizer_dataest(run_name, split, same_length=False, samples=None, base_results_dir="results", dups=0,
                                 only_new=False, cp_step=None):
    run_args = name_to_args(run_name)
    ec_type = run_args["ec_type"]
    lookup_len = run_args["lookup_len"]
    prequantization = run_args["prequantization"]
    n_hierarchical_clusters = run_args["n_hierarchical_clusters"]
    n_pca_components = run_args["n_pca_components"]
    n_clusters_pca = run_args["n_clusters_pca"]
    addec = run_args["addec"]
    alpha = run_args["alpha"]
    ec_source = run_args["ec_source"]
    if prequantization:
        from offline_quantizer import args_to_quant_dataset

        ecreact_dataset = args_to_quant_dataset(ec_type, n_hierarchical_clusters,
                                                n_pca_components, n_clusters_pca, alpha)
        ecreact_dataset = ecreact_dataset.replace("datasets/", "")
        if addec:
            ecreact_dataset += "_plus"

    else:
        ecreact_dataset = "ecreact/level4"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())

    if prequantization:
        from offline_quantizer import ResidualPCATokenizer

        new_tokens = ResidualPCATokenizer(n_residual_clusters=n_hierarchical_clusters,
                                          n_pca_components=n_pca_components,
                                          n_clusters_pca=n_clusters_pca,
                                          ).get_all_tokens()
        tokenizer.add_tokens(new_tokens)
    if ec_type == ECType.PAPER or addec:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    best_val_cp = get_best_val_cp(run_name, base_results_dir, cp_step)
    print("Loading model", best_val_cp)
    if (ec_type == ECType.PAPER or ec_type == ECType.NO_EC) or prequantization:
        model = T5ForConditionalGeneration.from_pretrained(best_val_cp)
    else:
        print("Loading custom model", best_val_cp)
        model = CustomT5Model.from_pretrained(best_val_cp, lookup_len=lookup_len)
    if same_length:
        max_length = args_to_lens(run_args)
    else:
        max_length = 200

    gen_dataset = SeqToSeqDataset([ecreact_dataset], split, tokenizer=tokenizer, ec_type=ec_type, DEBUG=False,
                                  save_ec=True, addec=addec, alpha=alpha, max_length=max_length,
                                  sample_size=samples, duplicated_source_mode=dups, ec_source=ec_source)
    if only_new:
        gen_dataset = get_only_new_ecs(gen_dataset)

    model.to(device)
    model.eval()
    return model, tokenizer, gen_dataset


def get_ec_from_df(dataset, per_level):
    all_ec = dataset.all_ecs
    if per_level != 0:
        all_ec = [" ".join(ec.strip().split(" ")[:per_level]) for ec in all_ec]
    else:
        all_ec = ["0"] * len(all_ec)
    return all_ec


def get_training_ec_count(level):
    with open("datasets/ecreact/level4/src-train.txt") as f:
        src_lines = f.read().splitlines()
    all_ec = [x.split("|")[1] for x in src_lines]
    all_ec = [" ".join(ec.strip().split(" ")[:level]) for ec in all_ec]
    return Counter(all_ec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="pretrained_5", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--fast", default=0, type=int)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--per_level", default=1, type=int)
    parser.add_argument("--per_ds", default=0, type=int)
    parser.add_argument("--dups", default=0, type=int)
    parser.add_argument("--res_base", default="results", type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--only_new", default=0, type=int)
    parser.add_argument("--cp_step", default=0, type=int)

    args = parser.parse_args()
    if args.cp_step == 0:
        args.cp_step = None
    run_name = args.run_name
    per_level = args.per_level

    print("---" * 10)
    print(f"Run: {run_name}")
    print("---" * 10)

    model, tokenizer, gen_dataset = load_model_tokenizer_dataest(run_name, args.split, dups=args.dups,
                                                                 base_results_dir=args.res_base, only_new=args.only_new,
                                                                 cp_step=args.cp_step)

    if args.per_ds:
        all_ec = gen_dataset.sources
    else:
        all_ec = get_ec_from_df(gen_dataset, per_level)

    gen_dataloader = DataLoader(gen_dataset, batch_size=args.bs, num_workers=0,
                                collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, model=model))

    # Evaluate the averaged model
    os.makedirs("results/full", exist_ok=True)
    with torch.no_grad():
        correct_count, ec_count = eval_dataset(model, tokenizer, gen_dataloader, k=args.k, fast=args.fast,
                                               save_file=f"results/full/{run_name}.csv", all_ec=all_ec)
    ec_training_count = get_training_ec_count(per_level)

    print(f"Run: {run_name}")
    for ec in correct_count:
        for i in range(1, args.k + 1):
            if ec_count[ec] == 0:
                continue
            print(f"{ec}: {i}: {correct_count[ec][i] / ec_count[ec]:.2f}")
        print(f"{ec}: {ec_count[ec]:.2f}")
    # Save the evaluation results
    output_file = f"results/eval_gen.csv"
    config_cols = run_name + "," + args.split + "," + str(args.k) + "," + str(args.fast) + "," + str(
        args.per_level) + "," + str(args.dups) + "," + str(args.only_new), str(args.per_ds), str(args.cp_step)
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(
                "run_name,split,k,fast,per_level,dups,only_new,per_ds,cp_step,rank,accuracy,ec_count,training_count\n")

    with open(output_file, "a") as f:  # Changed to append mode to log multiple runs
        for ec in correct_count:
            for i in range(1, args.k + 1):
                if ec_count[ec] == 0:
                    continue
                f.write(
                    config_cols + "," + ec + f",{i},{correct_count[ec][i] / ec_count[ec]:.4f},{ec_count[ec]},{ec_training_count[ec]}\n")
