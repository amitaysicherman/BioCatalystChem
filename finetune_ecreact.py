# sbatch --gres=gpu:1 --mem=16G --time=1-00:00:00 --account=def-bengioy --cpus-per-task=4 train.sh
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)
import os
from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments
import numpy as np
from rdkit import Chem

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


def load_pretrained_model():
    base_dir = "results/uspto"
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
                            n_clusters_pca, addec):
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
    pretrained_file = load_pretrained_model()
    pretrained_model = T5ForConditionalGeneration.from_pretrained(pretrained_file)
    pretrained_model.resize_token_embeddings(model.config.vocab_size)
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_model.state_dict(), strict=False)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Missing keys in the model (not loaded):", missing_keys)
    print("Unexpected keys in the checkpoint (not used by the model):", unexpected_keys)
    return tokenizer, model


def main(ec_type, lookup_len, prequantization, n_hierarchical_clusters, n_pca_components, n_clusters_pca, alpha, addec):
    ec_type = ECType(ec_type)
    tokenizer, model = get_tokenizer_and_model(ec_type, lookup_len, DEBUG, prequantization=prequantization,
                                               n_hierarchical_clusters=n_hierarchical_clusters,
                                               n_pca_components=n_pca_components, n_clusters_pca=n_clusters_pca,addec=addec)
    if prequantization:
        from offline_quantizer import args_to_quant_dataset
        ecreact_dataset = args_to_quant_dataset(ec_type, n_hierarchical_clusters,
                                                n_pca_components, n_clusters_pca, alpha)
        ecreact_dataset = ecreact_dataset.replace("datasets/", "")
        if addec:
            ecreact_dataset += "_plus"
    else:
        ecreact_dataset = "ecreact/level4"

    train_dataset = SeqToSeqDataset([ecreact_dataset,""], "uspto", weights=[100,1], tokenizer=tokenizer, ec_type=ec_type,
                                    DEBUG=DEBUG, alpha=alpha)
    train_small_dataset = SeqToSeqDataset([ecreact_dataset], "train", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                          DEBUG=DEBUG, sample_size=1000, alpha=alpha)
    val_small_dataset = SeqToSeqDataset([ecreact_dataset], "valid", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                        DEBUG=DEBUG, sample_size=1000, alpha=alpha)
    test_small_dataset = SeqToSeqDataset([ecreact_dataset], "test", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                         DEBUG=DEBUG, sample_size=1000, alpha=alpha)
    test_uspto_dataset = SeqToSeqDataset(["uspto"], "test", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                         DEBUG=DEBUG, sample_size=1000, alpha=alpha)

    eval_datasets = {"train": train_small_dataset, "valid": val_small_dataset, "test": test_small_dataset, "uspto": test_uspto_dataset}

    run_name = args_to_name(ec_type, lookup_len, prequantization, n_hierarchical_clusters, n_pca_components,
                            n_clusters_pca, alpha, addec)
    run_name += f"_mix"
    print(f"Run name: {run_name}")
    # Training arguments
    output_dir = f"results/{run_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        warmup_ratio=0.05,
        eval_steps=0.01,
        logging_steps=0.01,
        save_steps=0.01,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps",

        auto_find_batch_size=True,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256 // 8,
        eval_accumulation_steps=8,

        metric_for_best_model="eval_valid_accuracy",
        load_best_model_at_end=True,
        greater_is_better=True,
        report_to='none' if DEBUG else 'tensorboard',

        run_name=run_name,
        learning_rate=1e-5,

        save_safetensors=False

    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)


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
    args = parser.parse_args()
    args.alpha = float(args.alpha / 100)
    DEBUG = args.debug
    main(ec_type=args.ec_type, lookup_len=args.lookup_len, prequantization=args.prequantization,
         n_hierarchical_clusters=args.n_hierarchical_clusters, n_pca_components=args.n_pca_components,
         n_clusters_pca=args.n_clusters_pca, alpha=args.alpha, addec=args.addec)
