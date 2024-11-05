# sbatch --gres=gpu:A100:1 --mem=128G --time=7-00 --wrap="python train_uspto.py"
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
import numpy as np
from rdkit import Chem

from dataset import SeqToSeqDataset
from preprocessing.build_tokenizer import get_tokenizer_file_path
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

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


def get_tokenizer_and_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)
    if DEBUG:
        config.num_layers = 1
        config.d_model = 128
        config.num_heads = 4
        config.d_ff = 256
    model = T5ForConditionalGeneration(config)
    return tokenizer, model


def main(retro):
    tokenizer, model = get_tokenizer_and_model()

    train_dataset = SeqToSeqDataset(["uspto"], "train", weights=[1], tokenizer=tokenizer, DEBUG=DEBUG,retro=retro)
    eval_split = "valid" if not DEBUG else "train"
    train_small_dataset = SeqToSeqDataset(["uspto"], "train", weights=[1], tokenizer=tokenizer, DEBUG=DEBUG,
                                          sample_size=1000,retro=retro)
    val_small_dataset = SeqToSeqDataset(["uspto"], eval_split, weights=[1], tokenizer=tokenizer, DEBUG=DEBUG,
                                        sample_size=1000,retro=retro)
    test_small_dataset = SeqToSeqDataset(["uspto"], "test", weights=[1], tokenizer=tokenizer, DEBUG=DEBUG,
                                         sample_size=1000,retro=retro)
    eval_datasets = {"train": train_small_dataset, "valid": val_small_dataset, "test": test_small_dataset}
    run_name = "uspto_a40"
    output_dir = f"results/{run_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        warmup_ratio=0.05,
        eval_steps=0.01,
        logging_steps=0.01,
        save_steps=0.01,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps",

        auto_find_batch_size=True,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024 // 8,
        eval_accumulation_steps=8,

        metric_for_best_model="eval_valid_accuracy",
        load_best_model_at_end=True,
        greater_is_better=True,
        report_to='none' if DEBUG else 'tensorboard',

        run_name=run_name,
        learning_rate=1e-4,

        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )
    trainer.train()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--retro", action="store_true")

    args = parser.parse_args()
    DEBUG = args.debug
    main(args.retro)
