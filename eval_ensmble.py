import argparse
from torch.utils.data import DataLoader

import torch
import os
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from eval_gen import load_model_tokenizer_dataest, get_ec_from_df, tokens_to_canonical_smiles,eval_dataset

import torch
from tqdm import tqdm


def eval_ensemble(models, tokenizers, dataloaders,all_ecs):
    ens_scores = []
    for model, tokenizer, dataloader,ec in zip(models, tokenizers, dataloaders,all_ecs):
        correct_count, ec_count, all_scores=eval_dataset(model, tokenizer, dataloader, fast=1,all_ec=ec,return_all=True)
        ens_scores.append(all_scores)
    ens_scores = torch.stack(ens_scores, dim=1)
    ens_scores = ens_scores.mean(dim=1).mean(dim=0)
    print(f"Ensemble Score: {ens_scores.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_names", default=['dae-0.5_5', 'pretrained_5', 'paper'], type=str, nargs='+')
    parser.add_argument("--split", default="valid", type=str)
    parser.add_argument("--per_level", default=1, type=int)

    args = parser.parse_args()
    run_names = args.run_names
    per_level = args.per_level

    print("---" * 10)
    print(f"Run: {run_names}")
    print("---" * 10)
    models, tokenizers, dataloaders = [], [], []
    dataset_len = None
    all_ec = None
    for run_name in args.run_names:
        model, tokenizer, gen_dataset = load_model_tokenizer_dataest(run_name, args.split, same_length=True)
        models.append(model)
        tokenizers.append(tokenizer)
        if dataset_len is None:
            dataset_len = len(gen_dataset)
        else:
            assert dataset_len == len(gen_dataset)
        if all_ec is None:
            all_ec = get_ec_from_df(gen_dataset, per_level)
        else:
            assert all_ec == get_ec_from_df(gen_dataset, per_level)

        dataloaders.append(DataLoader(gen_dataset, batch_size=1, num_workers=0))

    # Evaluate the averaged model
    os.makedirs("results/full", exist_ok=True)
    with torch.no_grad():
        eval_ensemble(models, tokenizers, dataloaders,all_ecs=all_ec)
