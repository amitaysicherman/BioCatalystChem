# sbatch --gres=gpu:1 --mem=16G --time=1-00:00:00 --account=def-bengioy --cpus-per-task=4 train.sh
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)
from transformers import Trainer, TrainingArguments
import os
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file

import numpy as np
from rdkit import Chem

from dataset import SeqToSeqDataset, get_ec_type
from preprocessing.build_tokenizer import get_tokenizer_file_path, get_first_ec_token_index, get_ec_order
from model import CustomT5Model, EnzymaticT5Model
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import torch
from dataset import ECType
import re
import json

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


def args_to_name(use_ec, ec_split, lookup_len=5, dae=False, ecreact_only=0, freeze_encoder=0, post_encoder=0,
                 quantization=0
                 , q_groups=4, q_codevectors=512, q_index=0, prequantization=0,
                 n_hierarchical_clusters=5, n_pca_components=6, n_clusters_pca=10):
    if prequantization:
        suff = f"_{n_hierarchical_clusters}_{n_pca_components}_{n_clusters_pca}"
        if dae:
            run_name = f"prequantization_dae" + suff
        else:
            run_name = f"prequantization_pretrained" + suff
    elif dae:
        run_name = f"dae_{lookup_len}"
    elif use_ec:
        if ec_split:
            run_name = "paper"
        else:
            run_name = f"pretrained_{lookup_len}"
    else:
        run_name = "regular"
    if ecreact_only:
        run_name += "_ecreact"
    if freeze_encoder:
        run_name += "_freeze-enc"
    if post_encoder:
        run_name += "_post-enc"
    if quantization:
        index_suffix = "_inx" if q_index else ""
        run_name += "_quant_" + str(q_groups) + "_" + str(q_codevectors) + index_suffix
    return run_name


def get_tokenizer_and_model(ec_split, lookup_len, DEBUG=False, costum_t5=False, freeze_encoder=0, post_encoder=0,
                            quantization=0,
                            q_groups=5, q_codevectors=512, q_index=0, prequantization=0,
                            n_hierarchical_clusters=5, n_pca_components=6, n_clusters_pca=10):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())
    if prequantization:
        from offline_quantizer import HierarchicalPCATokenizer
        new_tokens = HierarchicalPCATokenizer(n_hierarchical_clusters=n_hierarchical_clusters,
                                              n_pca_components=n_pca_components,
                                              n_clusters_pca=n_clusters_pca,
                                              ).get_all_tokens()
        tokenizer.add_tokens(new_tokens)

    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)
    if DEBUG:
        config.num_layers = 1
        config.d_model = 128
        config.num_heads = 4
        config.d_ff = 256
    if prequantization or (ec_split and not costum_t5):
        model = T5ForConditionalGeneration(config)
        encoder = model.get_encoder()
    else:
        if not post_encoder:
            model = CustomT5Model(config, lookup_len, quantization=quantization, q_groups=q_groups,
                                  q_codevectors=q_codevectors, q_index=q_index)
            encoder = model.get_encoder()
        else:
            model = EnzymaticT5Model(config, lookup_len, quantization=quantization, q_groups=q_groups,
                                     q_codevectors=q_codevectors, q_index=q_index)
            encoder = model.t5_model.get_encoder()
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


def main(use_ec=True, ec_split=False, lookup_len=5, dae=False, load_cp="", ecreact_only=0, freeze_encoder=0,
         post_encoder=0, quantization=0, q_groups=5, q_codevectors=512, q_index=0, prequantization=0,
         n_hierarchical_clusters=5, n_pca_components=6, n_clusters_pca=10):
    tokenizer, model = get_tokenizer_and_model(ec_split, lookup_len, DEBUG, dae, freeze_encoder, post_encoder,
                                               quantization, q_groups=q_groups, q_codevectors=q_codevectors,
                                               q_index=q_index, prequantization=prequantization,
                                               n_hierarchical_clusters=n_hierarchical_clusters,
                                               n_pca_components=n_pca_components, n_clusters_pca=n_clusters_pca)
    if load_cp:
        if load_cp=="best_no_ecreact":
            run_name_no_eceract = args_to_name(use_ec, ec_split, lookup_len, dae, 0, freeze_encoder, post_encoder,
                                    quantization,
                                    q_groups, q_codevectors, q_index, prequantization, n_hierarchical_clusters,
                                    n_pca_components,
                                    n_clusters_pca)
            load_cp=f"results/{run_name_no_eceract}"
            cp_dirs = os.listdir(load_cp)
            cp_dirs = [f for f in cp_dirs if re.match(r"checkpoint-\d+", f)]
            cp_dirs = sorted(cp_dirs, key=lambda x: int(x.split("-")[1]))
            last_cp = f"{load_cp}/{cp_dirs[-1]}"
            trainer_state_file = f"{last_cp}/trainer_state.json"
            with open(trainer_state_file) as f:
                trainer_state = json.load(f)
            load_cp = trainer_state["best_model_checkpoint"]
        model_filename="model.safetensors"
        if os.path.exists(os.path.join(load_cp, model_filename)):
            loaded_state_dict =load_file(os.path.join(load_cp, model_filename))
        else:
            model_filename = "pytorch_model.bin"
            loaded_state_dict = torch.load(os.path.join(load_cp, model_filename), map_location="cpu")

        if prequantization:
            m = model.t5_model if isinstance(model, EnzymaticT5Model) else model
            model_v_size = len(m.shared.weight)
            cp_v_size = len(loaded_state_dict["shared.weight"])
            if model_v_size!=cp_v_size:
                d_model = m.config.d_model
                random_init_new_tokens_param = torch.randn(model_v_size - cp_v_size, d_model)
                new_shared = torch.cat([loaded_state_dict["shared.weight"], random_init_new_tokens_param], dim=0)
                loaded_state_dict["shared.weight"] = new_shared.float()
        if isinstance(model, EnzymaticT5Model):
            missing_keys, unexpected_keys = model.t5_model.load_state_dict(loaded_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
        print("Missing keys in the model (not loaded):", missing_keys)
        print("Unexpected keys in the checkpoint (not used by the model):", unexpected_keys)

    # ecreact_dataset = "ecreact/level3" if ec_split else "ecreact/level4"

    ec_type = get_ec_type(use_ec, ec_split, dae) if not prequantization else ECType.PAPER
    if prequantization:
        from offline_quantizer import args_to_quant_dataset
        ecreact_dataset = args_to_quant_dataset(ECType.DAE if dae else ECType.PRETRAINED, n_hierarchical_clusters,
                                                n_pca_components, n_clusters_pca)
        ecreact_dataset = ecreact_dataset.replace("datasets/", "")

    else:
        ecreact_dataset = "ecreact/level4"
    if ecreact_only:
        train_datasets_names = [ecreact_dataset]
        w = [1]
    else:
        train_datasets_names = [ecreact_dataset, "uspto"]
        w = [9, 1]

    train_dataset = SeqToSeqDataset(train_datasets_names, "train", weights=w, tokenizer=tokenizer,
                                    ec_type=ec_type, DEBUG=DEBUG)
    eval_split = "valid" if not DEBUG else "train"

    train_ecreact_small = SeqToSeqDataset([ecreact_dataset], "train", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                          DEBUG=DEBUG, sample_size=3000)
    val_ecreact = SeqToSeqDataset([ecreact_dataset], eval_split, weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                  DEBUG=DEBUG)
    test_ecreact = SeqToSeqDataset([ecreact_dataset], "test", weights=[1], tokenizer=tokenizer, ec_type=ec_type,
                                   DEBUG=DEBUG)
    eval_datasets = {"ecreact": val_ecreact, "ecreact_train": train_ecreact_small, "ecreact_test": test_ecreact}

    run_name = args_to_name(use_ec, ec_split, lookup_len, dae, ecreact_only, freeze_encoder, post_encoder, quantization,
                            q_groups, q_codevectors, q_index, prequantization, n_hierarchical_clusters,
                            n_pca_components,
                            n_clusters_pca)
    print(f"Run name: {run_name}")
    # Training arguments
    output_dir = f"results/{run_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_steps=5_000 if not DEBUG else 10,
        save_total_limit=2,
        max_steps=100_000,
        # auto_find_batch_size=True,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64 // 8,
        logging_steps=2_500 if not DEBUG else 10,
        eval_steps=5_000 if not DEBUG else 10,
        metric_for_best_model="eval_ecreact_accuracy",
        warmup_steps=8_000 if not DEBUG else 10,
        eval_accumulation_steps=8,
        report_to='none' if DEBUG else 'tensorboard',
        run_name=run_name,
        resume_from_checkpoint=True,
        load_best_model_at_end=True,
        learning_rate=1e-4,
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
    # Train the model
    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_ec", default=1, type=int)
    parser.add_argument("--ec_split", default=0, type=int)
    parser.add_argument("--dae", default=0, type=int)
    parser.add_argument("--lookup_len", default=5, type=int)
    parser.add_argument("--load_cp", default="", type=str)
    parser.add_argument("--ecreact_only", default=0, type=int)
    parser.add_argument("--freeze_encoder", default=0, type=int)
    parser.add_argument("--post_encoder", default=0, type=int)
    parser.add_argument("--quantization", default=0, type=int)
    parser.add_argument("--q_groups", default=4, type=int)
    parser.add_argument("--q_codevectors", default=512, type=int)
    parser.add_argument("--q_index", default=0, type=int)
    parser.add_argument("--prequantization", default=0, type=int)
    parser.add_argument("--n_hierarchical_clusters", type=int, default=5)
    parser.add_argument("--n_pca_components", type=int, default=6)
    parser.add_argument("--n_clusters_pca", type=int, default=10)

    args = parser.parse_args()
    DEBUG = args.debug
    main(args.use_ec, args.ec_split, args.lookup_len, args.dae, args.load_cp, args.ecreact_only,
         freeze_encoder=args.freeze_encoder, post_encoder=args.post_encoder, quantization=args.quantization,
         q_groups=args.q_groups, q_codevectors=args.q_codevectors, q_index=args.q_index,
         prequantization=args.prequantization, n_hierarchical_clusters=args.n_hierarchical_clusters,
         n_pca_components=args.n_pca_components, n_clusters_pca=args.n_clusters_pca)
