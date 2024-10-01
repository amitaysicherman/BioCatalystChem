import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
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
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.prot_dim = 2560
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D").eval().to(device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)



    def fasta_to_vec(self, seq: str):
        assert seq != ""
        inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
        inputs = clip_to_max_len(inputs)
        with torch.no_grad():
            vec = self.model(inputs)['last_hidden_state'][0]
        return vec.detach().cpu().numpy()


if __name__ == "__main__":
    ec2vec = EC2Vec()
    ec_mapping = pd.read_csv("datasets/ec_map.csv")
    uniprot_to_fasta = defaultdict(str)
    for i, row in ec_mapping.iterrows():
        uniprot_to_fasta[row["Uniprot_id"]] = row["Sequence"]
    base_dir = "datasets/docking"
    all_dirs = os.listdir(base_dir)
    for uniprot_id in tqdm(all_dirs):
        fasta = uniprot_to_fasta[uniprot_id]
        vec = ec2vec.fasta_to_vec(fasta)
        np.save(f"{base_dir}/{uniprot_id}/protein.npy", vec)
