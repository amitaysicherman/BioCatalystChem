import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
import re
from rdkit import Chem
import glob
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from preprocessing.build_tokenizer import redo_ec_split
from tqdm import tqdm
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")
EPSILON = 1e-6
aa3to1 = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
    'MSE': 'M',
}


def get_protein_cords(pdb_file):
    ca_pattern = re.compile(r"^ATOM\s+\d+\s+CA\s+([A-Z]{3})\s+([\w])\s+\d+\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)")
    seq = []
    cords = []
    with open(pdb_file, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith("ENDMDL"):
                break
            match = ca_pattern.match(line)
            if match:
                resn = match.group(1)  # Residue name (e.g., SER)
                chain = match.group(2)  # Chain identifier (e.g., A)
                assert chain == "A"  # For esmfold from fasta
                x_coord = float(match.group(3))  # X coordinate
                y_coord = float(match.group(4))  # Y coordinate
                z_coord = float(match.group(5))  # Z coordinate
                seq.append(aa3to1.get(resn, 'X'))
                cords.append([x_coord, y_coord, z_coord])
    return "".join(seq), cords


def get_mol_cords(sdf_file):
    coords = []
    supplier = Chem.SDMolSupplier(sdf_file)

    # Iterate through the molecules and extract coordinates
    for mol in supplier:
        if mol is None:
            continue  # skip invalid molecules
        conf = mol.GetConformer()  # get the conformation
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
    return coords


def check_protein_exists(protein_id):
    protein_file = f'datasets/pdb_files/{protein_id}/{protein_id}_esmfold.pdb'
    protein_emd_file = f'datasets/docking/{protein_id}/protein.npy'
    return os.path.exists(protein_file) and os.path.exists(protein_emd_file)


def calculate_dsw(distances, vdw_products=1.7, clip_value=1.91):
    distances = np.clip(distances, clip_value, None)
    return (1 / distances) * (2 * np.power(vdw_products / distances, 12) - np.power(vdw_products / distances, 6))


def get_protein_mol_att(protein_id, molecule_id):
    protein_file = f'datasets/pdb_files/{protein_id}/{protein_id}_esmfold.pdb'
    protein_seq, protein_cords = get_protein_cords(protein_file)
    protein_cords = np.array(protein_cords)
    ligand_dir = f'datasets/docking2/{protein_id}/{molecule_id}/complex_0/'
    sdf_files = glob.glob(f"{ligand_dir}*.sdf")

    all_weights = []  # Store coordinates from all .sdf files
    for sdf_file in sdf_files:
        lig_coords = get_mol_cords(sdf_file)
        if len(lig_coords) > 0:
            dist = euclidean_distances(protein_cords, lig_coords)
            weights = calculate_dsw(dist)
            weights = weights.sum(axis=1)
            weights = weights / weights.sum()
            all_weights.append(weights)
    if not all_weights:
        return None  # If no coordinates were found, return None
    weights = np.array(all_weights).mean(axis=0)
    return weights
    # protein_emd_file = f'datasets/docking/{protein_id}/protein.npy'
    # emb = np.load(protein_emd_file)[1:-1]  # remove cls and eos tokens
    # if len(emb) != len(weights):
    #     print(f"Length mismatch: {len(emb)} vs {len(weights)}")
    #     return None
    # docking_attention_emd = np.average(emb, axis=0, weights=weights)
    # if return_weights:
    #     return docking_attention_emd, weights
    # return docking_attention_emd


def get_reaction_attention_emd(non_can_smiles, ec, ec_to_uniprot, smiles_to_id):
    protein_id = ec_to_uniprot[ec]
    if not check_protein_exists(protein_id):
        return None
    weights = []
    non_can_smiles = non_can_smiles.replace(" ", "")
    for s in non_can_smiles.split("."):

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        Chem.RemoveStereochemistry(mol)
        s = Chem.MolToSmiles(mol)

        if s in smiles_to_id:
            ligand_file = f'datasets/docking/{protein_id}/{smiles_to_id[s]}/complex_0/rank1.sdf'
            if not os.path.exists(ligand_file):
                continue
            molecule_id = smiles_to_id[s]
            try:
                w = get_protein_mol_att(protein_id, molecule_id)
            except:
                continue
            if w is not None:
                weights.append(w)
    if len(weights) == 0:
        return None
    weights = np.array(weights).mean(axis=0)
    # add 0,0 in the beginning and end for cls and eos tokens
    weights = np.concatenate([[0], weights, [0]])
    return weights


def args_to_file(v2):
    docking_dir = "docking2" if v2 else "docking"
    return "datasets/" + docking_dir + f"/docking.npz"


def load_docking_file(v2):
    print("Loading docking file")
    d = np.load(args_to_file(v2))
    src_ec_to_vec = dict()

    for key in tqdm(d.keys()):
        src, ec = key.split("|")
        src_ec_to_vec[(src, ec)] = d[key]
    return src_ec_to_vec


class Docker:
    def __init__(self, v2=1):
        self.docker = load_docking_file(v2)

    def dock_src_ec(self, src, ec):
        key = (src, ec)
        if key in self.docker:
            return self.docker[key]
        return None

    def dock_src_line(self, line):
        src, ec = redo_ec_split(line, True)
        return self.dock_src_ec(src, ec)


if __name__ == "__main__":
    docker = Docker(1)
    with open("datasets/ecreact/level4/src-train.txt") as f:
        src_lines = f.read().splitlines()
    for text in src_lines[:10]:
        docker.dock_src_line(text)
