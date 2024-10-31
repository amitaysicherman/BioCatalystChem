import os
import pandas as pd
from rdkit import Chem
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")



dir_path = os.path.join('datasets', 'ecreact')
os.makedirs(dir_path, exist_ok=True)

url = "https://raw.githubusercontent.com/rxn4chemistry/biocatalysis-model/main/data/ecreact-1.0.csv"
file_path = os.path.join(dir_path, 'ecreact-1.0.csv')
if not os.path.isfile(file_path):
    os.system(f"curl -o {file_path} {url}")
os.path.isfile(file_path), file_path

#
# def remove_stereochemistry_from_smiles(input_smiles):
#     molecules = input_smiles.split(".")
#     molecules = [Chem.MolFromSmiles(x) for x in molecules]
#     for mol in molecules:
#         Chem.RemoveStereochemistry(mol)
#     return ".".join([Chem.MolToSmiles(x) for x in molecules])
#
#
# def remove_stereochemistry(rxn_smiles):
#     src_ec, tgt = rxn_smiles.split(">>")
#     src, ec = src_ec.split("|")
#     src = remove_stereochemistry_from_smiles(src)
#     tgt = remove_stereochemistry_from_smiles(tgt)
#     return f"{src}|{ec}>>{tgt}"


output_file = os.path.join(dir_path, 'ecreact-1.0.txt')
df = pd.read_csv(file_path)
txt_file = []
for i in range(len(df)):
    txt_file.append(df['rxn_smiles'][i])
with open(output_file, 'w') as f:
    for item in txt_file:
        # item = remove_stereochemistry(item)
        f.write("%s\n" % item)

prep_script = "python preprocessing/biocatalysis-model/bin/rbt-preprocess.py"

# level 3
# output_dir = os.path.join(dir_path, 'level3')
# os.makedirs(output_dir, exist_ok=True)
# os.system(f"{prep_script} {output_file} {output_dir} --ec-level 3")

# level 4
output_dir = os.path.join(dir_path, 'level4')
os.makedirs(output_dir, exist_ok=True)
os.system(f"{prep_script} {output_file} {output_dir} --ec-level 4")
