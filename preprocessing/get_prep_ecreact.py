import os
import pandas as pd

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
        f.write("%s\n" % item)

prep_script = "python preprocessing/biocatalysis-model/bin/rbt-preprocess.py"
os.system(f"{prep_script} {output_file} {dir_path} ")
