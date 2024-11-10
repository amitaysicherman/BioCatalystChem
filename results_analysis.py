from eval_gen import eval_dataset, load_model_tokenizer_dataest, get_ec_from_df
import torch

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="pretrained_5", type=str)
    parser.add_argument("--fast", default=1, type=int)
    args = parser.parse_args()

    run_name = args.run_name
    per_level = 1

    print("---" * 10)
    print(f"Run: {run_name}")
    print("---" * 10)
    splits = ["train", "valid", "test"]
    res = dict()
    model, tokenizer, [train_df, valid_ds, test_df] = load_model_tokenizer_dataest(run_name, splits)
    for split, dataset in zip(splits, [train_df, valid_ds, test_df]):
        split_ec = get_ec_from_df(dataset, per_level)
        print(f"Split: {split}")
        with torch.no_grad():
            correct_count, ec_count = eval_dataset(model, tokenizer, dataset, all_ec=split_ec, k=1, fast=1,
                                                   save_file=None)
        res[split] = {f"[v{i}]": correct_count[i][1] / ec_count[i] for i in correct_count}
        print(res[split])
