from tqdm import tqdm
from dataclasses import dataclass
from bioservices import UniProt
import os
import pandas as pd
from typing import List
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class Protein:
    ec_full: str
    ec_use: str
    uniprot_id: str
    seq: str
    pdb_id: str
    pdb_file: str


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


def protein_to_dataframes(proteins: List[Protein]):
    data = []
    for protein in proteins:
        data.append([protein.ec_full, protein.ec_use, protein.seq, protein.pdb_id, protein.pdb_file])
    df = pd.DataFrame(data, columns=["EC_full", "EC_use", "Uniprot_id", "Sequence", "PDB_ID", "PDB_file"])
    return df


def process_ec(ec: str):
    id_, fasta, ec_use = ec_to_id_fasta(ec)
    if id_ == "":
        return None
    return Protein(ec, ec_use, id_, fasta, "", "")


if __name__ == "__main__":
    uniprot = UniProt()
    output_dir = "datasets/pdb_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_dataset = "datasets/ecreact/ecreact-1.0.txt"
    with open(base_dataset) as f:
        lines = f.readlines()

    all_ec = set()
    for line in lines:
        ec = line.split("|")[1].split(">>")[0]
        all_ec.add(ec)

    proteins = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU cores
        futures = {executor.submit(process_ec, ec): ec for ec in all_ec}
        for future in tqdm(as_completed(futures), total=len(futures)):
            protein = future.result()
            if protein:
                proteins.append(protein)

    df = protein_to_dataframes(proteins)
    df.to_csv(f"datasets/ec_map.csv", index=False)

    protein_ligend_for_pdbs = [(p.uniprot_id, p.seq, "O") for p in proteins]
    df = pd.DataFrame(protein_ligend_for_pdbs, columns=["complex_name", "protein_sequence", "ligand_description"])

    output_pdb_dir = "datasets/pdb_files"
    n_splits = 8
    for i in range(n_splits):
        df_split = df.iloc[i::n_splits]
        df_split.to_csv(f"{output_pdb_dir}/protein_ligand_{i}.csv", index=False)
