from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration
import argparse
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset import SeqToSeqDataset
from preprocessing.build_tokenizer import get_tokenizer_file_path, get_first_ec_token_index, get_ec_order
from model import CustomT5Model
import torch
import os
import re
from tqdm import tqdm
from rdkit import RDLogger

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

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=200, do_sample=False, num_beams=k * 2,
                                 num_return_sequences=k)
        mask = (labels != tokenizer.pad_token_id) & (labels != -100)
        labels = labels[mask]

        labels = tokens_to_canonical_smiles(tokenizer, labels)
        preds_list = [tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs]
        for j in range(1, k + 1):
            if labels in preds_list[:j]:
                correct_count[j] += 1
        msg = " | ".join([f"{j}:{correct_count[j]/(i+1):.2f}" for j in range(1, k + 1)])
        msg = f'{i+1}/{len(gen_dataloader)} | {msg}'
        pbar.set_description(msg)
    return {i: correct_count[i] / len(gen_dataloader) for i in [1, 3, 5, 10]}


def average_checkpoints(cp_dirs, model_type, models_args):
    """
    Averages the state_dicts of all checkpoints provided in cp_dirs.
    """
    if not cp_dirs:
        raise ValueError("No checkpoints found to average.")

    # Initialize the averaged state_dict
    avg_state_dict = None
    num_checkpoints = len(cp_dirs)

    for idx, cp_dir in enumerate(cp_dirs):
        print(f"Loading checkpoint {idx + 1}/{num_checkpoints}: {cp_dir}")
        model = model_type.from_pretrained(cp_dir, **models_args)
        state_dict = model.state_dict()

        if avg_state_dict is None:
            # Initialize with the first checkpoint's state_dict
            avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            # Accumulate the parameters
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float()

        # Free up memory
        del model
        torch.cuda.empty_cache()

    # Compute the average
    for k in avg_state_dict:
        avg_state_dict[k] /= num_checkpoints

    return avg_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="pretrained_5", type=str)
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
        models_args = {
            "lookup_len": lookup_len,
            "ec_tokens_order": get_ec_order(tokenizer, ec_split),
            "cutoff_index": get_first_ec_token_index(tokenizer, ec_split)
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

    gen_dataset = SeqToSeqDataset([dataset], "valid", tokenizer=tokenizer, use_ec=use_ec,
                                  ec_split=ec_split, DEBUG=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=1, num_workers=0)

    # List and sort all checkpoint directories
    cp_dirs = sorted(
        [f for f in os.listdir(cp_dir_all) if re.match(r"checkpoint-\d+", f)],
        key=lambda x: int(x.split("-")[1])
    )
    cp_dirs = [f"{cp_dir_all}/{cp_dir}" for cp_dir in cp_dirs]

    if not cp_dirs:
        raise ValueError(f"No checkpoints found in {cp_dir_all}")

    # Average all checkpoints
    averaged_state_dict = average_checkpoints(cp_dirs, model_type, models_args)

    # Initialize the model and load the averaged state_dict
    if "pretrained" in run_name:
        averaged_model = model_type.from_pretrained(cp_dirs[0], **models_args)
    else:
        averaged_model = model_type.from_pretrained(cp_dirs[0], **models_args)

    averaged_model.load_state_dict(averaged_state_dict)
    averaged_model.to(device)
    averaged_model.eval()

    # Evaluate the averaged model
    correct_count = eval_dataset(averaged_model, gen_dataloader)
    print(f"Run: {run_name} (Averaged Checkpoints)")
    for k, acc in correct_count.items():
        print(f"{k}: {acc}")

    # Save the evaluation results
    output_file = f"results/eval_gen.csv"
    with open(output_file, "a") as f:  # Changed to append mode to log multiple runs
        f.write(run_name + ",averaged," + ",".join([str(correct_count[i]) for i in [1, 3, 5, 10]]) + "\n")
