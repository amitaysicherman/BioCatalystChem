from tqdm import tqdm
from bioservices import UniProt
import pandas as pd
import torch

def ec_to_id_fasta(ec: str):
    while "." in ec:
        results = uniprot.search(f"ec:{ec}", limit=1, frmt='fasta', size=1)
        results = results.splitlines()
        if len(results) < 2:
            ec = ".".join(ec.split(".")[:-1])
        else:
            break
    if len(results) < 2:
        return "", "", ""
    id_ = results[0].split("|")[1]
    fasta = "".join(results[1:])
    return id_, fasta, ec


if __name__ == "__main__":
    uniprot = UniProt()
    base_dataset = "datasets/ecreact/ecreact-1.0.txt"
    with open(base_dataset) as f:
        lines = f.readlines()
    results = []
    mem={}
    pbar=tqdm(lines, total=len(lines))
    for line in pbar:
        ec = line.split("|")[1].split(">>")[0]
        if ec not in mem:
            mem[ec]=ec_to_id_fasta(ec)
        id_, fasta, ec_use = mem[ec]
        input_smiles = line.split("|")[0].split(".")
        for smile in input_smiles:
            name=f"{id_}${smile}"
            results.append([name, "", smile, fasta])
        pbar.set_description("Total pairs: %d" % len(results))
    df = pd.DataFrame(results, columns=["complex_name", "protein_path", "ligand_description", "protein_sequence"])
    df.to_csv("datasets/protein_ligand.csv", index=False)