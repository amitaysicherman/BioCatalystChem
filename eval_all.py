import os
import json
from train import get_tokenizer_and_model, compute_metrics
from safetensors.torch import load_file
import torch
from dataset import SeqToSeqDataset
from transformers import PreTrainedTokenizerFast
from preprocessing.build_tokenizer import get_tokenizer_file_path
from dataset import ECType, get_ec_type
from transformers import Trainer, TrainingArguments
import pandas as pd


def name_to_args(run_name):
    """
    Inverse function of args_to_name that converts a run name string back to its arguments

    Args:
        run_name (str): Name string generated by args_to_name

    Returns:
        dict: Dictionary containing all parsed arguments
    """
    args = {
        'use_ec': False,
        'ec_split': True,
        'lookup_len': 5,
        'dae': False,
        'ecreact_only': 0,
        'freeze_encoder': 0,
        'post_encoder': 0,
        'quantization': 0,
        'q_groups': 4,
        'q_codevectors': 512,
        'q_index': 0,
        'costum_t5': False
    }

    # Parse base run type
    if run_name.startswith("dae_"):
        args['dae'] = True
        args['lookup_len'] = int(run_name.split('_')[1].split('_')[0])
        args['costum_t5'] = True
    elif run_name.startswith("paper"):
        args['use_ec'] = True
        args['ec_split'] = True
    elif run_name.startswith("pretrained_"):
        args['use_ec'] = True
        args['lookup_len'] = int(run_name.split('_')[1].split('_')[0])
        args['ec_split'] = False
    elif run_name.startswith("regular"):
        pass  # default values are already set
    if "_ecreact" in run_name:
        args['ecreact_only'] = 1
    if "_freeze-enc" in run_name:
        args['freeze_encoder'] = 1
    if "_post-enc" in run_name:
        args['post_encoder'] = 1
    if "_quant" in run_name:
        args['quantization'] = 1
        if run_name.endswith("_quant"):
            args['q_groups'] = 4
            args['q_codevectors'] = 512
        else:
            quant_parts = run_name.split("_quant_")[1].split("_")
            args['q_groups'] = int(quant_parts[0])
            args['q_codevectors'] = int(quant_parts[1])
            if len(quant_parts) > 2 and quant_parts[2].endswith("inx"):
                args['q_index'] = 1

    return args


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


DEBUG = False
tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())
datasets_per_ec = {}
for ec_type in [ECType.DAE, ECType.PRETRAINED, ECType.PAPER, ECType.NO_EC]:
    ec_datasets = dict()
    for part in ['train', 'valid', 'test']:
        if part == "train":
            sample_size = 3000
        else:
            sample_size = None
        ec_datasets[part] = SeqToSeqDataset(['ecreact/level4'], part, tokenizer=tokenizer,
                                            ec_type=ec_type, DEBUG=DEBUG, sample_size=sample_size)
    datasets_per_ec[ec_type] = ec_datasets
results = []
for run_name in os.listdir("results"):
    try:
        args = name_to_args(run_name)
        best_cp = get_best_val_cp(run_name)

        tokenizer, model = get_tokenizer_and_model(ec_split=args['ec_split'], lookup_len=args['lookup_len'],
                                                   freeze_encoder=args['freeze_encoder'],
                                                   post_encoder=args['post_encoder'], costum_t5=args['costum_t5'],
                                                   quantization=args['quantization'], q_groups=args['q_groups'],
                                                   q_codevectors=args['q_codevectors'], q_index=args['q_index'])
        if os.path.exists(best_cp + "/model.safetensors"):
            loaded_state_dict = load_file(best_cp + "/" + "model.safetensors")
        else:
            model_file = "pytorch_model.bin"
            loaded_state_dict = torch.load(best_cp + "/" + model_file)
        model.load_state_dict(loaded_state_dict)
        model.eval()

        ec_type = get_ec_type(args['use_ec'], args['ec_split'], dae=args['dae'])
        datasets = datasets_per_ec[ec_type]
        trainer_args = TrainingArguments(
            report_to="none", output_dir=run_name + "/tmp"
        )
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=datasets,
            eval_dataset=datasets,
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metrics(x, tokenizer)
        )
        scores = trainer.evaluate()
        scores['run_name'] = run_name
        results.append(scores)

    except Exception as e:
        print(f"Error in {run_name}")
        print(args)
        print(best_cp)
        print(e)
        continue

results = pd.DataFrame(results)
results.to_csv("results/results.csv")
