# sbatch --gres=gpu:1 --mem=16G --time=1-00:00:00 --account=def-bengioy --cpus-per-task=4 train.sh
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)
import os
from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

import numpy as np
from rdkit import Chem
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    PeftConfig
)
import torch

from dataset import SeqToSeqDataset
from preprocessing.build_tokenizer import get_tokenizer_file_path, get_ec_tokens
from model import CustomT5Model
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from dataset import ECType
import json
import os
import re

DEBUG = False
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")


def compute_metrics(eval_pred, tokenizer):
    predictions_, labels_ = eval_pred
    predictions_ = np.argmax(predictions_[0], axis=-1)
    token_acc = []
    accuracy = []
    is_valid = []
    for i in range(len(predictions_)):
        mask = (labels_[i] != tokenizer.pad_token_id) & (labels_[i] != -100)
        pred = predictions_[i][mask]
        label = labels_[i][mask]
        token_acc.append((pred == label).mean().item())
        pred = tokenizer.decode(pred, skip_special_tokens=True)
        is_valid.append(Chem.MolFromSmiles(pred.replace(" ", "")) is not None)
        label = tokenizer.decode(label, skip_special_tokens=True)
        accuracy.append(pred == label)

    token_acc = np.mean(token_acc)
    accuracy = np.mean(accuracy)
    is_valid = np.mean(is_valid)
    return {"accuracy": accuracy, "valid_smiles": is_valid, "token_acc": token_acc}


def args_to_name(ec_type, lookup_len, prequantization, n_hierarchical_clusters, n_pca_components, n_clusters_pca,
                 alpha, addec):
    if ec_type == ECType.PAPER:
        return "paper"
    elif ec_type == ECType.NO_EC:
        return "regular"
    elif ec_type == ECType.PRETRAINED:
        run_name = f"pretrained"
    elif ec_type == ECType.DAE:
        run_name = f"dae-{alpha}"
    else:
        raise ValueError(f"Invalid ec_type: {ec_type}")

    if prequantization:
        suff = f"_{n_hierarchical_clusters}_{n_pca_components}_{n_clusters_pca}"
        run_name += "_quant" + suff
    else:
        run_name += f"_{lookup_len}"
    if addec:
        run_name += "_plus"
    return run_name


def load_pretrained_model(regpre):
    if not regpre:
        base_dir = "results/uspto"
    else:
        base_dir = "results/regular"

    cp_dirs = os.listdir(base_dir)
    cp_dirs = [f for f in cp_dirs if re.match(r"checkpoint-\d+", f)]
    cp_dirs = sorted(cp_dirs, key=lambda x: int(x.split("-")[1]))
    last_cp = f"{base_dir}/{cp_dirs[-1]}"
    trainer_state_file = f"{last_cp}/trainer_state.json"
    if not os.path.exists(trainer_state_file):
        raise ValueError(f"trainer_state.json not found in {base_dir}")
    with open(trainer_state_file) as f:
        trainer_state = json.load(f)
    return trainer_state["best_model_checkpoint"]


def get_tokenizer_and_model(ec_type, lookup_len, DEBUG, prequantization, n_hierarchical_clusters, n_pca_components,
                            n_clusters_pca, addec, nopre, lora, lora_d, regpre):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())
    if prequantization:
        from offline_quantizer import HierarchicalPCATokenizer
        new_tokens = HierarchicalPCATokenizer(n_hierarchical_clusters=n_hierarchical_clusters,
                                              n_pca_components=n_pca_components,
                                              n_clusters_pca=n_clusters_pca,
                                              ).get_all_tokens()
        tokenizer.add_tokens(new_tokens)
    if ec_type == ECType.PAPER or addec:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)
    if DEBUG:
        config.num_layers = 1
        config.d_model = 128
        config.num_heads = 4
        config.d_ff = 256
    if (ec_type == ECType.PAPER or ec_type == ec_type.NO_EC) or prequantization:
        model = T5ForConditionalGeneration(config)
    else:
        model = CustomT5Model(config, lookup_len)
    if not nopre:
        pretrained_file = load_pretrained_model(regpre=regpre)
        pretrained_model = T5ForConditionalGeneration.from_pretrained(pretrained_file)

        pretrained_model.resize_token_embeddings(model.config.vocab_size)
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Missing keys in the model (not loaded):", missing_keys)
    if lora:
        modules_to_save = ["shared"]
        target_modules = [key for key, _ in model.named_modules() if "Linear" in str(type(_))]

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_d,
            lora_alpha=lora_d,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return tokenizer, model


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        if "emb" not in features[0]:
            return super().__call__(features)
        emb = [f.pop("emb") for f in features]
        batch = super().__call__(features)
        batch["emb"] = torch.stack(emb)  # Stack the 'emb' tensors into a batch
        return batch


def main(ec_type, lookup_len, prequantization, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, addec,
         nopre, lora, lora_d, regpre, mix, batch_size=256, learning_rate=1e-3):
    if DEBUG:
        batch_size = 8
    ec_type = ECType(ec_type)
    tokenizer, model = get_tokenizer_and_model(ec_type, lookup_len, DEBUG, prequantization=prequantization,
                                               n_hierarchical_clusters=n_hierarchical_clusters,
                                               n_pca_components=n_pca_components, n_clusters_pca=n_clusters_pca,
                                               addec=addec, nopre=nopre, lora=lora, lora_d=lora_d, regpre=regpre)
    if prequantization:
        from offline_quantizer import args_to_quant_dataset
        ecreact_dataset = args_to_quant_dataset(ec_type, n_hierarchical_clusters,
                                                n_pca_components, n_clusters_pca, alpha)
        ecreact_dataset = ecreact_dataset.replace("datasets/", "")
        if addec:
            ecreact_dataset += "_plus"
    else:
        ecreact_dataset = "ecreact/level4"

    if mix:
        train_dataset = SeqToSeqDataset([ecreact_dataset, "uspto"], "train", weights=[20, 1], tokenizer=tokenizer,
                                        ec_type=ec_type, DEBUG=DEBUG, alpha=alpha, addec=addec)
    else:
        train_dataset = SeqToSeqDataset([ecreact_dataset], "train", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                        DEBUG=DEBUG, alpha=alpha, addec=addec)
    train_small_dataset = SeqToSeqDataset([ecreact_dataset], "train", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                          DEBUG=DEBUG, sample_size=1000, alpha=alpha, addec=addec)
    val_small_dataset = SeqToSeqDataset([ecreact_dataset], "valid", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                        DEBUG=DEBUG, alpha=alpha, addec=addec)
    test_small_dataset = SeqToSeqDataset([ecreact_dataset], "test", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                         DEBUG=DEBUG, sample_size=1000, alpha=alpha, addec=addec)
    test_uspto_dataset = SeqToSeqDataset(["uspto"], "test", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                         DEBUG=DEBUG, alpha=alpha, addec=addec)

    eval_datasets = {"train": train_small_dataset, "valid": val_small_dataset, "test": test_small_dataset,
                     "uspto": test_uspto_dataset}

    run_name = args_to_name(ec_type, lookup_len, prequantization, n_hierarchical_clusters, n_pca_components,
                            n_clusters_pca, alpha, addec)
    if nopre:
        run_name += f"_nopre"
    if regpre:
        run_name += f"_regpre"
    if lora:
        run_name += f"_lora_{lora_d}"
    if mix:
        run_name += f"_mix"
    run_name += f"_bs-{batch_size}_lr-{learning_rate}"
    print(f"Run name: {run_name}")
    # Training arguments
    output_dir = f"results/{run_name}"
    if not mix:
        num_train_epochs = 100
    else:
        num_train_epochs = 5
    if batch_size > 16:
        gradient_accumulation_steps = batch_size // 16
        batch_size = 16
    else:
        gradient_accumulation_steps = 1

    dirs_in_output = os.listdir(output_dir)
    # check if there is a checkpoint to resume from
    resume_from_checkpoint = False
    for dir in dirs_in_output:
        if "checkpoint" in dir:
            resume_from_checkpoint = True
            break
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.05,
        eval_steps=0.05,
        logging_steps=0.05,
        save_steps=0.05,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps",

        auto_find_batch_size=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 8,
        eval_accumulation_steps=8,

        metric_for_best_model="eval_valid_accuracy",
        load_best_model_at_end=True,
        greater_is_better=True,
        report_to='none' if DEBUG else 'tensorboard',

        run_name=run_name,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_safetensors=False,
        group_by_length=True,
        resume_from_checkpoint=resume_from_checkpoint
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ec_type", default=1, type=int)
    parser.add_argument("--lookup_len", default=5, type=int)
    parser.add_argument("--prequantization", default=0, type=int)
    parser.add_argument("--n_hierarchical_clusters", type=int, default=5)
    parser.add_argument("--n_pca_components", type=int, default=6)
    parser.add_argument("--n_clusters_pca", type=int, default=10)
    parser.add_argument("--alpha", type=int, default=50)
    parser.add_argument("--addec", type=int, default=0)
    parser.add_argument("--nopre", type=int, default=0)
    parser.add_argument("--regpre", type=int, default=0)
    parser.add_argument("--mix", type=int, default=0)
    parser.add_argument("--lora", type=int, default=0)
    parser.add_argument("--lora_d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--tasks_on_gpu", type=int, default=1)
    args = parser.parse_args()
    if args.tasks_on_gpu > 1:
        torch.cuda.set_per_process_memory_fraction(1 / args.tasks_on_gpu)

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)  # in GB
    reserved_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)  # in GB
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Allocated VRAM by this process: {allocated_vram:.2f} GB")
    print(f"Reserved VRAM by this process: {reserved_vram:.2f} GB")

    args.alpha = float(args.alpha / 100)
    DEBUG = args.debug
    main(ec_type=args.ec_type, lookup_len=args.lookup_len, prequantization=args.prequantization,
         n_hierarchical_clusters=args.n_hierarchical_clusters, n_pca_components=args.n_pca_components,
         n_clusters_pca=args.n_clusters_pca, alpha=args.alpha, addec=args.addec, nopre=args.nopre, lora=args.lora,
         lora_d=args.lora_d, regpre=args.regpre, mix=args.mix, batch_size=args.batch_size,
         learning_rate=args.learning_rate)
