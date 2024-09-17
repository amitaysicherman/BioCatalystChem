import os
output_dir = "datasets/uspto"
os.makedirs(output_dir, exist_ok=True)

src_test = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-test.txt"
src_train_1 = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-train-split-1.txt"
src_train_2 = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-train-split-2.txt"
src_valid = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-valid.txt"
tgt_test = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/tgt-test.txt"
tgt_train = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/tgt-train.txt"
tgt_valid = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/tgt-valid.txt"

for url in [src_test, src_train_1, src_train_2, src_valid, tgt_test, tgt_train, tgt_valid]:
    file_path = os.path.join(output_dir, os.path.basename(url))
    if not os.path.isfile(file_path):
        os.system(f"curl -o {file_path} {url}")

# combine train splits
lines = []
src_split_1 = os.path.join(output_dir, 'src-train-split-1.txt')
src_split_2 = os.path.join(output_dir, 'src-train-split-2.txt')
output_file = os.path.join(output_dir, 'src-train.txt')
with open(src_split_1) as f:
    lines.extend(f.readlines())
with open(src_split_2) as f:
    lines.extend(f.readlines())
with open(output_file, 'w') as f:
    f.writelines(lines)

os.remove(src_split_1)
os.remove(src_split_2)
