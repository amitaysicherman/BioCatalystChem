import pandas as pd
from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration, T5Config
import argparse
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset_v2 import SeqToSeqDataset
from preprocessing.build_tokenizer import get_tokenizer_file_path
from model import CustomT5Model
import torch
import os
from tqdm import tqdm
from rdkit import RDLogger
from dataset import ECType
from preprocessing.build_tokenizer import get_ec_tokens
from finetune_ecreact_v2 import CustomDataCollatorForSeq2Seq
import json

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def name_to_args(name):
    name = name.replace("_nmix", "")
    if "NO_EC" in name:
        return {"ec_type": ECType.NO_EC, "daa_type": 0, "add_ec": 0, "add_mode": 0}
    add_mode = 0
    if "add" in name:
        add_mode = 1
        name = name.replace("_add", "")
    add_ec = 0
    if "ec" in name:
        add_ec = 1
        name = name.replace("_ec", "")
    parts = name.split("_")
    daa_type = 0
    if len(parts) > 1:
        daa_type = parts[1]
    ec_type_name = parts[0]
    ec_type = ECType[ec_type_name]
    return {"ec_type": ec_type, "daa_type": int(daa_type), "add_ec": add_ec, "add_mode": add_mode}


def tokens_to_canonical_smiles(tokenizer, tokens, remove_stereo=False):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def k_name(filename, k):
    assert filename.endswith(".txt")
    return filename.replace(".txt", f"_k{k}.txt")


def eval_dataset(model, tokenizer, dataloader, output_file, all_k=[1, 3, 5], remove_stereo=False):
    k = max(all_k)
    k_to_res = {k_: [] for k_ in all_k}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_ids = batch['id'].detach().cpu().numpy().flatten().tolist()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device).bool()
        labels = batch['labels'].to(model.device)
        emb = batch['emb'].to(model.device).float()
        scores = batch['docking_scores'].to(model.device).float()
        emb_mask = batch['emb_mask'].to(model.device).bool()
        if (emb == 0).all():
            emb_args = {}
        else:
            emb_args = {"emb": emb, "emb_mask": emb_mask, "docking_scores": scores}

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=200, do_sample=False, num_beams=k,
                                 num_return_sequences=k, **emb_args)

        for j in range(len(batch_ids)):
            mask = (labels[j] != tokenizer.pad_token_id) & (labels[j] != -100)
            label = labels[j][mask]
            label_smiles = tokens_to_canonical_smiles(tokenizer, label, remove_stereo=remove_stereo)
            preds_list = [tokens_to_canonical_smiles(tokenizer, opt, remove_stereo=remove_stereo) for opt in
                          outputs[j * k:(j + 1) * k]]
            id_ = batch_ids[j]
            for k_ in all_k:
                is_correct = int(label_smiles in preds_list[:k_])
                k_to_res[k_].append((id_, is_correct))
    for k_ in all_k:
        with open(k_name(output_file, k_), "w") as f:
            for id_, is_correct in k_to_res[k_]:
                f.write(f"{id_},{is_correct}\n")


def get_cp_epoch(cp):
    state_file = f"{cp}/trainer_state.json"
    with open(state_file) as f:
        state = json.load(f)
    return float(state["epoch"])


def get_all_cp(base_dir):
    all_cp = os.listdir(base_dir)
    all_cp = [f for f in all_cp if f.startswith("checkpoint-")]
    all_cp = sorted(all_cp, key=lambda x: int(x.split("-")[1]))

    all_cp = [f"{base_dir}/{cp}" for cp in all_cp]
    cp_epochs = [get_cp_epoch(cp) for cp in all_cp]
    return all_cp, cp_epochs


def load_model_tokenizer_dataest(run_name, split, base_results_dir, sample_size):
    run_args = name_to_args(run_name)
    ec_type = run_args["ec_type"]
    daa_type = run_args["daa_type"]
    addec = run_args["add_ec"]
    add_mode = run_args["add_mode"]

    ecreact_dataset = "ecreact/level4"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())
    if ec_type == ECType.PAPER or addec:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    all_cps, cp_steps = get_all_cp(f"{base_results_dir}/{run_name}")
    models = []
    for cp in all_cps:
        print("Loading model", cp)
        if (ec_type == ECType.PAPER or ec_type == ECType.NO_EC):
            model = T5ForConditionalGeneration.from_pretrained(cp)
        else:
            print("Loading custom model", cp)
            config = T5Config.from_pretrained(cp)
            model = CustomT5Model(config, daa_type=daa_type, add_mode=add_mode)
            model.load_state_dict(torch.load(f"{cp}/pytorch_model.bin"))
        model.to(device)
        model.eval()
        models.append(model)

    if ec_type == ECType.PAPER or ec_type == ECType.NO_EC:
        add_emb = False
    else:
        add_emb = True
    gen_dataset = SeqToSeqDataset([ecreact_dataset], split, tokenizer=tokenizer, max_length=200, add_emb=[add_emb],
                                  sample_size=sample_size)

    return models, tokenizer, gen_dataset, cp_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="PAPER", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--res_base", default="results", type=str)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--sample_size", default=0, type=int)
    parser.add_argument("--remove_stereo", default=0, type=int)
    parser.add_argument("--last_only", default=1, type=int)
    args = parser.parse_args()
    run_name = args.run_name
    sample_size = args.sample_size if args.sample_size > 0 else None
    print("---" * 10)
    print(f"Run: {run_name}")
    print("---" * 10)
    models, tokenizer, gen_dataset, cp_steps = load_model_tokenizer_dataest(run_name, args.split,
                                                                            base_results_dir=args.res_base,
                                                                            sample_size=sample_size)
    if args.last_only:
        models = [models[-1]]
        cp_steps = [cp_steps[-1]]
    gen_dataloader = DataLoader(gen_dataset, batch_size=args.bs, num_workers=0,
                                collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, model=models[0]))

    with torch.no_grad():
        for j, model in enumerate(models):
            if args.remove_stereo:
                output_file = f"results/{run_name}/rs_{args.split}_{cp_steps[j]}.txt"
            else:
                output_file = f"results/{run_name}/{args.split}_{cp_steps[j]}.txt"

            eval_dataset(model, tokenizer, gen_dataloader, all_k=[1, 3, 5], output_file=output_file,
                         remove_stereo=args.remove_stereo)
