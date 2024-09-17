import torch
import re
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, BertForMaskedLM, BertTokenizer, EsmModel
import numpy as np
from bioservices import UniProt
import os
from tqdm import tqdm

MAX_LEN = 510
PROTEIN_MAX_LEN = 1023

P_BFD = "bfd"
P_T5_XL = "t5"
ESM_1B = "ems1"
ESM_2 = "esm2"

protein_name_to_cp = {
    P_BFD: 'Rostlab/prot_bert_bfd',
    P_T5_XL: 'Rostlab/prot_t5_xl_half_uniref50-enc',
    ESM_1B: 'facebook/esm1b_t33_650M_UR50S',
    ESM_2: 'facebook/esm2_t36_3B_UR50D',
}

model_to_dim = {
    P_BFD: 1024,
    P_T5_XL: 1024,
    ESM_1B: 1280,
    ESM_2: 2560
}

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
    def __init__(self, name=ESM_2):
        super().__init__()
        self.cp_name = protein_name_to_cp[name]
        self.tokenizer = None
        self.model = None
        self.name = name
        self.get_model_tokenizer()
        self.prot_dim = model_to_dim[name]
        self.random_if_fail = False
        self.uniprot = UniProt()
        self.ec_to_vec_mem = dict()
        self.ec_to_id = dict()
        self.id_to_ec = dict()
        self.ec_to_vec_file = f"datasets/{name}_ec_to_vec.txt"
        self.load_ec_to_vec()

    def load_ec_to_vec(self):
        if os.path.exists(self.ec_to_vec_file):
            with open(self.ec_to_vec_file) as f:
                for line in f:
                    id_, ec, vec = line.strip().split("\t")
                    self.ec_to_vec_mem[ec] = np.fromstring(vec, sep=" ")
                    self.ec_to_id[ec] = id_
                    self.id_to_ec[id_] = ec

    def save_ec_to_vec(self):
        with open(self.ec_to_vec_file, "w") as f:
            for ec, vec in self.ec_to_vec_mem.items():
                vec_str = " ".join(map(str, vec))
                f.write(f"{self.ec_to_id[ec]}\t{ec}\t{vec_str}\n")

    def add_ec_to_vec(self, ec, vec):
        with open(self.ec_to_vec_file, "a") as f:
            vec_str = " ".join(map(str, vec))
            f.write(f"{len(self.ec_to_vec_mem)}\t{ec}\t{vec_str}\n")

    def post_process(self, vec):
        return vec.detach().cpu().numpy().flatten()

    def get_model_tokenizer(self):
        if self.name == P_BFD:
            self.tokenizer = BertTokenizer.from_pretrained(self.cp_name, do_lower_case=False)
            self.model = BertForMaskedLM.from_pretrained(self.cp_name, output_hidden_states=True).eval().to(device)
        elif self.name == ESM_1B or self.name == ESM_2:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cp_name)
            self.model = EsmModel.from_pretrained(self.cp_name).eval().to(device)
        elif self.name == P_T5_XL:
            self.tokenizer = T5Tokenizer.from_pretrained(self.cp_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(self.cp_name).eval().to(device)
        else:
            raise ValueError(f"Unknown protein embedding: {self.name}")
        if device == torch.device("cpu"):
            self.model.to(torch.float32)

    def fasta_to_vec(self, seq: str):
        if seq == "":
            if self.random_if_fail:
                return np.random.rand(1, self.prot_dim)
            return torch.zeros(1, self.prot_dim)
        if self.name in [ESM_1B, ESM_2]:
            inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
            inputs = clip_to_max_len(inputs)
            with torch.no_grad():
                vec = self.model(inputs)['pooler_output'][0]
        else:
            seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
            ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            input_ids = clip_to_max_len(input_ids, PROTEIN_MAX_LEN)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            attention_mask = clip_to_max_len(attention_mask, PROTEIN_MAX_LEN)

            with torch.no_grad():
                embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if self.name == P_BFD:
                vec = embedding_repr.hidden_states[-1][0].mean(dim=0)
            else:
                vec = embedding_repr.last_hidden_state[0].mean(dim=0)
        self.prot_dim = vec.shape[-1]
        return self.post_process(vec)

    def ec_to_fasta(self, ec: str):
        result = self.uniprot.search(f"ec:{ec}", limit=1, frmt='fasta', size=1)
        return "".join(result.splitlines()[1:])

    def ec_to_vec(self, ec: str):
        if ec not in self.ec_to_vec_mem:
            fasta = self.ec_to_fasta(ec)
            self.ec_to_vec_mem[ec] = self.fasta_to_vec(fasta)
            self.add_ec_to_vec(ec, self.ec_to_vec_mem[ec])
        return self.ec_to_vec_mem[ec]

    def id_to_vec(self, id_: str):
        return self.ec_to_vec(self.id_to_ec[id_])

    def ids_to_vecs(self, ids: torch.Tensor):
        return torch.stack([self.id_to_vec(id_) for id_ in ids])


if __name__ == "__main__":
    ec2vec = EC2Vec()
    base_dataset = "datasets/ecreact/ecreact-1.0.txt"
    with open(base_dataset) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        ec = line.split("|")[1].split(">>")[0]
        ec2vec.ec_to_vec(ec)
