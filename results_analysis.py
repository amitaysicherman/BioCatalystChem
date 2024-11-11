from eval_gen import eval_dataset, load_model_tokenizer_dataest, get_ec_from_df
import torch
from torch.utils.data import DataLoader
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
    res = dict()
    model, tokenizer, [train_df, valid_ds, test_df] = load_model_tokenizer_dataest(run_name, splits,samples=[None, None, 1000])
    for split, dataset in zip(splits, [train_df, valid_ds, test_df]):
        split_ec = get_ec_from_df(dataset, per_level)
        gen_dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        print(f"Split: {split}")
        with torch.no_grad():
            correct_count, ec_count = eval_dataset(model, tokenizer, dataset, all_ec=split_ec, k=1, fast=1,
                                                   save_file=None)
        res[split] = {f"[v{i}]": correct_count[i][1] / ec_count[i] for i in correct_count}
        print(res[split])

    for key in res["test"]:
        print(f"{key}: Train: {res['train'][key]:.2f}, Valid: {res['valid'][key]:.2f}, Test: {res['test'][key]:.2f}")
