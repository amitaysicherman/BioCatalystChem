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

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokens_to_canonical_smiles(tokenizer, tokens):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def eval_dataset(model: T5ForConditionalGeneration, gen_dataloader: DataLoader, k=10):
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
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=200, do_sample=False, num_beams=k * 2,
                                 num_return_sequences=k, **emb_args)
        mask = (labels != tokenizer.pad_token_id) & (labels != -100)
        labels = labels[mask]

        labels = tokens_to_canonical_smiles(tokenizer, labels)
        preds_list = [tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs]
        for j in range(1, k + 1):
            if labels in preds_list[:j]:
                correct_count[j] += 1
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
    parser.add_argument("--run_name", default="pretrained_5_seq", type=str)
    parser.add_argument("--split", default="test", type=str)
    args = parser.parse_args()
    dataset = "ecreact/level4"
    run_name = args.run_name
    best_val_cp = get_best_val_cp(run_name)
    if "pretrained" in run_name or "dae" in run_name:
        if "pretrained" in run_name:
            ec_split = False
            use_ec = True
        else:  # dae
            ec_split = True
            use_ec = True

        lookup_len = int(run_name.split("_")[1])
        tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path(ec_split))
        model_type = CustomT5Model
        models_args = {
            "lookup_len": lookup_len,
            "seq_or_add": 1 if "add" in run_name else 0
        }

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

    ec_type = get_ec_type(use_ec, ec_split, 'dae' in run_name)
    gen_dataset = SeqToSeqDataset([dataset], args.split, tokenizer=tokenizer, ec_type=ec_type, DEBUG=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=1, num_workers=0)

    # List and sort all checkpoint directories
    model = model_type.from_pretrained(best_val_cp, **models_args)
    model.to(device)
    model.eval()

    # Evaluate the averaged model
    correct_count = eval_dataset(model, gen_dataloader)
    print(f"Run: {run_name} (Averaged Checkpoints)")
    for k, acc in correct_count.items():
        print(f"{k}: {acc}")

    # Save the evaluation results
    output_file = f"results/eval_gen.csv"
    with open(output_file, "a") as f:  # Changed to append mode to log multiple runs
        f.write(run_name + args.split + "," + best_val_cp + "," + ",".join(
            [str(correct_count[i]) for i in [1, 3, 5, 10]]) + "\n")
