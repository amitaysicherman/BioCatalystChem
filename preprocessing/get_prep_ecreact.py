import os
import pandas as pd
import re
import numpy as np

dir_path = os.path.join('datasets', 'ecreact')
os.makedirs(dir_path, exist_ok=True)

url = "https://raw.githubusercontent.com/rxn4chemistry/biocatalysis-model/main/data/ecreact-1.0.csv"
file_path = os.path.join(dir_path, 'ecreact-1.0.csv')
if not os.path.isfile(file_path):
    os.system(f"curl -o {file_path} {url}")
os.path.isfile(file_path), file_path

output_file = os.path.join(dir_path, 'ecreact-1.0.txt')
df = pd.read_csv(file_path)
txt_file = []
for i in range(len(df)):
    txt_file.append(df['rxn_smiles'][i])
with open(output_file, 'w') as f:
    for item in txt_file:
        # item = remove_stereochemistry(item)
        f.write("%s\n" % item)

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_TOKENIZER_PATTERN)


def tokenize_enzymatic_reaction_smiles(rxn: str) -> str:
    parts = re.split(r">|\|", rxn)
    ec = parts[1].split(".")
    if len(ec) != 4:
        print(f"Error: {rxn} ({len(ec)})")
        return None
    rxn = rxn.replace(f"|{parts[1]}", "")
    tokens = [token for token in SMILES_REGEX.findall(rxn)]
    arrow_index = tokens.index(">>")

    levels = ["v", "u", "t", "q"]

    if ec[0] != "":
        ec_tokens = [f"[{levels[i]}{e}]" for i, e in enumerate(ec)]
        ec_tokens.insert(0, "|")
        tokens[arrow_index:arrow_index] = ec_tokens

    return " ".join(tokens)


output_tokenized_file = os.path.join(dir_path, 'ecreact-1.0-tokenized.txt')
tokens_lines = []
src = []
tgt = []
datasets = []
ecreact = pd.read_csv(file_path)
with open(output_tokenized_file, 'w') as f2:
    for i, row in ecreact.iterrows():
        rnx = row['rxn_smiles']
        source = row['source']
        tokens = tokenize_enzymatic_reaction_smiles(rnx)
        if tokens:
            src_, tgt_ = tokens.split(' >> ')
            src.append(src_)
            tgt.append(tgt_)
            datasets.append(source)
            f2.write(tokens + '\n')

print(f"Tokenized {len(src)} reactions")
assert len(src) == len(tgt)
# split into train val test 70 15 15
np.random.seed(42)
indices = np.random.permutation(len(src))
train_indices = indices[:int(0.7 * len(src))]
val_indices = indices[int(0.7 * len(src)):int(0.85 * len(src))]
test_indices = indices[int(0.85 * len(src)):]
for split, split_indices in zip(['train', 'valid', 'test'], [train_indices, val_indices, test_indices]):
    split_src = [src[i] for i in split_indices]
    split_tgt = [tgt[i] for i in split_indices]
    split_dst = [datasets[i] for i in split_indices]
    print(f"{split}: {len(split_src)}")
    with open(os.path.join(dir_path, f'level4/src-{split}.txt'), 'w') as f:
        for line in split_src:
            f.write(f"{line}\n")
    with open(os.path.join(dir_path, f'level4/tgt-{split}.txt'), 'w') as f:
        for line in split_tgt:
            f.write(f"{line}\n")
    with open(os.path.join(dir_path, f'level4/datasets-{split}.txt'), 'w') as f:
        for line in split_dst:
            f.write(f"{line}\n")