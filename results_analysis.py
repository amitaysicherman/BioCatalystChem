import pandas as pd

from eval_gen import eval_dataset, load_model_tokenizer_dataest, get_ec_from_df
from finetune_ecreact import CustomDataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from rdkit import Chem

ALL_EC = ["[v1]", "[v2]", "[v3]", "[v4]", "[v5]", "[v6]", "[v7]"]


def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def predict_batch(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizerFast, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    labels = batch['labels']
    emb_args = {"emb": batch['emb']} if not (batch['emb'] == 0).all() else {}
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels, **emb_args)
    predictions = outputs.logits.argmax(dim=-1)
    mask = (labels != tokenizer.pad_token_id) & (labels != -100)
    pred = predictions[mask]
    label = labels[mask]
    pred = tokenizer.decode(pred, skip_special_tokens=True).replace(" ", "")
    label = tokenizer.decode(label, skip_special_tokens=True).replace(" ", "")
    return pred, label


def slice_batch(batch, slice_token=237, step=0):
    input_ids = batch['input_ids'].squeeze(0)  # Shape: (seq_len,)
    attention_mask = batch['attention_mask'].squeeze(0)  # Shape: (seq_len,)

    # Find the index of the token "237" in the sequence
    try:
        token_position = (input_ids == slice_token).nonzero()[0][0].item() + step
    except IndexError:
        # If token "237" is not found, you can handle it here
        token_position = input_ids.size(0)  # Set to end of sequence if "237" not found

    # Slice input_ids and attention_mask up to the token position
    input_ids_sliced = input_ids[:token_position].unsqueeze(0)  # Shape: (1, seq_len)
    attention_mask_sliced = attention_mask[:token_position].unsqueeze(0)  # Shape: (1, seq_len)
    batch_sliced = {
        'input_ids': input_ids_sliced,
        'attention_mask': attention_mask_sliced,
        'labels': batch['labels'].clone(),
        'emb': batch['emb'].clone()
    }
    return batch_sliced


def eval_dataset(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizerFast, gen_dataloader: DataLoader,
                 all_ec):
    counts = {ec: 0 for ec in all_ec}
    correct_count = {ec: 0 for ec in all_ec}
    valid_smiles = {ec: 0 for ec in all_ec}
    correct_ec_0 = {ec: 0 for ec in all_ec}
    correct_ec_1 = {ec: 0 for ec in all_ec}
    correct_ec_2 = {ec: 0 for ec in all_ec}
    correct_ec_3 = {ec: 0 for ec in all_ec}

    no_ec_list = [correct_ec_0, correct_ec_1, correct_ec_2, correct_ec_3]
    for i, batch in tqdm(enumerate(gen_dataloader), total=len(gen_dataloader)):
        pred, label = predict_batch(model, tokenizer, batch)
        counts[all_ec[i]] += 1
        correct_count[all_ec[i]] += int(pred == label)
        valid_smiles[all_ec[i]] += int(is_valid_smiles(pred))
        for step, correct_ec in enumerate(no_ec_list):
            sliced_batch = slice_batch(batch, step=step)
            pred_sliced, label_sliced = predict_batch(model, tokenizer, sliced_batch)
            correct_ec[all_ec[i]] += int(pred_sliced == label_sliced)
        # if pred != label and is_valid_smiles(pred):
        #     print(f"Real: {label} VS Predicted: {pred}")
        #     real_mol = Chem.MolFromSmiles(label)
        #     pred_mol = Chem.MolFromSmiles(pred)
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        #     ax1.imshow(Chem.Draw.MolToImage(real_mol))
        #     ax1.set_title("Real Molecule")
        #     ax2.imshow(Chem.Draw.MolToImage(pred_mol))
        #     ax2.set_title("Predicted Molecule")
        #     plt.show()

    return correct_count, counts, valid_smiles, *no_ec_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="regular_mix_bs-256_lr-0.001", type=str)
    args = parser.parse_args()

    run_name = args.run_name
    per_level = 1

    print("---" * 10)
    print(f"Run: {run_name}")
    print("---" * 10)
    splits = ["train", "valid", "test"]
    model, tokenizer, [train_df, valid_ds, test_df] = load_model_tokenizer_dataest(run_name, splits,
                                                                                   samples=[1000, None, None])
    res = dict()
    scores_types = ["correct_count", "counts", "valid_smiles", "correct_no_ec_0", "correct_no_ec_1", "correct_no_ec_2",
                    "correct_no_ec_3"]
    multi_level_columns = pd.MultiIndex.from_product([splits, scores_types], names=["Split", "Scores"])
    results = pd.DataFrame(columns=multi_level_columns, index=ALL_EC)

    for split, dataset in zip(splits, [train_df, valid_ds, test_df]):
        split_ec = get_ec_from_df(dataset, per_level)
        gen_dataloader = DataLoader(dataset, batch_size=1, num_workers=0,
                                    collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, model=model))

        print(f"Split: {split}")
        with torch.no_grad():
            correct_count, counts, valid_smiles, *no_ec_list = eval_dataset(model, tokenizer, gen_dataloader,
                                                                            all_ec=split_ec)
        res[split] = {i: correct_count[i] / counts[i] for i in correct_count}
        smiles_res = {i: valid_smiles[i] / counts[i] for i in valid_smiles}
        for i, correct_no_ec in enumerate(no_ec_list):
            no_ec_list[i] = {i: correct_no_ec[i] / counts[i] for i in correct_no_ec}
        for ec in ALL_EC:
            if split == "test" and ec == "[v7]":
                continue
            results.loc[ec, (split, "correct_count")] = res[split][ec]
            results.loc[ec, (split, "counts")] = counts[ec]
            results.loc[ec, (split, "valid_smiles")] = smiles_res[ec]
            results.loc[ec, (split, "correct_no_ec_0")] = no_ec_list[0][ec]
            results.loc[ec, (split, "correct_no_ec_1")] = no_ec_list[1][ec]
            results.loc[ec, (split, "correct_no_ec_2")] = no_ec_list[2][ec]
            results.loc[ec, (split, "correct_no_ec_3")] = no_ec_list[3][ec]
        print(results)
        print("---" * 10)
        print(results.to_csv())
