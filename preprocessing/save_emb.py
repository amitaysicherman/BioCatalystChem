# sbatch --mem=64G --time=7-00 --gres=gpu:A40:1 ----requeue --wrap "python preprocessing/save_emb.py"
import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from esm.sdk import client
from esm.sdk.api import LogitsConfig, ESMProtein
import time
MAX_LEN = 510
PROTEIN_MAX_LEN = 1023

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    try:
        torch.mps.empty_cache()
        device = torch.device('mps')
    except:
        device = torch.device('cpu')


def clip_to_max_len(x: torch.Tensor, max_len: int = 1023):
    if x.shape[1] <= max_len:
        return x
    last_token = x[:, -1:]
    clipped_x = x[:, :max_len - 1]
    result = torch.cat([clipped_x, last_token], dim=1)
    return result


class EC2Vec:
    def __init__(self, name="esm2"):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.prot_dim = 2560
        self.name = name
        if name == "esm2":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            self.model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D").eval().to(device)
            if device == torch.device("cpu"):
                self.model.to(torch.float32)
        elif name == "esm3":
            self.model = client("esm3-medium-2024-08", token="3hn8PHelb0F4FdWgrLxXKR")
        else:
            raise ValueError(f"Invalid model name: {name}")

    def fasta_to_vec(self, seq: str):
        assert seq != ""
        if self.name == "esm2":
            inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
            inputs = clip_to_max_len(inputs)
            with torch.no_grad():
                vec = self.model(inputs)['last_hidden_state'][0]
        elif self.name == "esm3":
            try:
                protein = ESMProtein(sequence=seq)
                protein = self.model.encode(protein)
                conf = LogitsConfig(return_embeddings=True, sequence=True)
                vec = self.model.logits(protein, conf).embeddings[0]
            except Exception as e:
                print(f"Error: {e}")
                print(f"Sequence: {seq}")
                print("Sleeping for 60 seconds")
                time.sleep(60)
                protein = ESMProtein(sequence=seq)
                protein = self.model.encode(protein)
                conf = LogitsConfig(return_embeddings=True, sequence=True)
                vec = self.model.logits(protein, conf).embeddings[0].mean(dim=0)

        else:
            raise ValueError(f"Invalid model name: {self.name}")
        return vec.detach().cpu().numpy()


if __name__ == "__main__":
    name = "esm3"
    ec2vec = EC2Vec(name=name)
    ec_mapping = pd.read_csv("datasets/ec_map.csv")
    uniprot_to_fasta = defaultdict(str)
    for i, row in ec_mapping.iterrows():
        uniprot_to_fasta[row["Uniprot_id"]] = row["Sequence"]
    base_dir = "datasets/docking"
    all_dirs = os.listdir(base_dir)
    all_dirs = [x for x in all_dirs if os.path.isdir(f"{base_dir}/{x}")]
    for uniprot_id in tqdm(all_dirs):
        fasta = uniprot_to_fasta[uniprot_id]
        vec = ec2vec.fasta_to_vec(fasta)
        if name == "esm2":
            np.save(f"{base_dir}/{uniprot_id}/protein.npy", vec)
        elif name == "esm3":
            np.save(f"{base_dir}/{uniprot_id}/ems3_protein.npy", vec)
        else:
            raise ValueError(f"Invalid model name: {name}")


