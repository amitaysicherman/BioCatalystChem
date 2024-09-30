from tqdm import tqdm
from dataclasses import dataclass
from bioservices import UniProt
import os
import requests
import pandas as pd
from typing import List
from transformers import AutoTokenizer, EsmForProteinFolding
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs




class ESMFold:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.esm = self.model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        self.chunk_size = 256
        self.model.trunk.set_chunk_size(self.chunk_size)

    def fold(self, seq, output_file):
        tokenized_input = self.tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)

        output = None

        while output is None:
            try:
                with torch.no_grad():
                    output = self.model(tokenized_input)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on chunk_size', self.chunk_size)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    self.chunk_size = self.chunk_size // 2
                    if self.chunk_size > 2:
                        self.model.set_chunk_size(self.chunk_size)
                    else:
                        print("Not enough memory for ESMFold")
                        break
                else:
                    raise e
        if output is None:
            return
        pdb_file = convert_outputs_to_pdb(output)[0]
        with open(output_file, "w") as f:
            f.write(pdb_file)


@dataclass
class Protein:
    ec_full: str
    ec_use: str
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


def uniprot_id_to_pdb_id(uniprot_id: str):
    url = "https://rest.uniprot.org/uniprotkb/search?query="
    url += uniprot_id
    url += "&fields=structure_3d"
    results = requests.get(url).json()
    if 'results' not in results:
        return ""
    results = results['results']
    if len(results) == 0 or 'uniProtKBCrossReferences' not in results[0] or len(
            results[0]['uniProtKBCrossReferences']) == 0:
        return ""
    return results[0]['uniProtKBCrossReferences'][0]['id']


def get_pdb_file(pdb_id: str, output_dir: str):
    if pdb_id == "":
        return ""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_file = f"{output_dir}/{pdb_id}.pdb"
    cmd = f"wget {url} -O {output_file} -q"
    res = os.system(cmd)
    if res != 0:
        return ""
    return f"{output_dir}/{pdb_id}.pdb"


def protein_to_dataframes(proteins: List[Protein]):
    data = []
    for protein in proteins:
        data.append([protein.ec_full, protein.ec_use, protein.seq, protein.pdb_id, protein.pdb_file])
    df = pd.DataFrame(data, columns=["EC_full", "EC_use", "Sequence", "PDB_ID", "PDB_file"])
    return df


if __name__ == "__main__":
    uniprot = UniProt()
    esm_fold = ESMFold()
    output_dir = "datasets/pdb_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_dataset = "datasets/ecreact/ecreact-1.0.txt"
    with open(base_dataset) as f:
        lines = f.readlines()
    all_ec = set()
    proteins = []
    for line in lines:
        ec = line.split("|")[1].split(">>")[0]
        all_ec.add(ec)
    for ec in tqdm(all_ec):
        id_, fasta, ec_use = ec_to_id_fasta(ec)
        if id_ == "":
            continue
        pdb_id = uniprot_id_to_pdb_id(id_)
        if pdb_id != "":
            pdb_file = get_pdb_file(pdb_id, output_dir)
        else:
            pdb_file = ""
        if pdb_file == "":
            pdb_file = f"{output_dir}/{id_}.pdb"
            esm_fold.fold(fasta, pdb_file)
        proteins.append(Protein(ec, ec_use, fasta, pdb_id, pdb_file))
    df = protein_to_dataframes(proteins)
    df.to_csv(f"datasets/ec_map.csv", index=False)
