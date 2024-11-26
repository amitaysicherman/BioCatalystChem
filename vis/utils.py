from collections import defaultdict

import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
import numpy as np


def get_residue_ids_from_pdb(pdb_file):
    """
    Extracts residue IDs from a PDB file in the order they appear in the file.
    Returns a list of residue IDs that correspond to the protein sequence.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # We use the residue ID as a tuple of (chain ID, residue sequence number, insertion code)
                residue_id = (chain.id, residue.id[1])
                residue_ids.append(residue_id)
    return residue_ids


def replace_local_pathes(script_path):
    with open(script_path) as f:
        c = f.read()
    c = c.replace("datasets", "/Users/amitay.s/PycharmProjects/BioCatalystChem/datasets")
    with open(script_path, "w") as f:
        f.write(c)


def remove_stereo(s):
    s = s.replace(" ", "")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    print(Chem.MolToSmiles(mol))
    return Chem.MolToSmiles(mol)


def load_maps():
    with open("datasets/docking/smiles_to_id.txt") as f:
        id_to_smile = {int(x.split()[1]): x.split()[0] for x in f.readlines()}
        smile_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
    ec_mapping = pd.read_csv("datasets/ec_map.csv")
    uniport_to_ec = defaultdict(str)
    for i, row in ec_mapping.iterrows():
        uniport_to_ec[row["Uniprot_id"]] = row["EC_full"]
    return id_to_smile, smile_to_id, uniport_to_ec


def remove_dup_mis_mols(molecules_ids, id_to_smile):
    molecules_smiles = [id_to_smile[int(x)] for x in molecules_ids]
    molecules_smiles_no_stereo = [remove_stereo(x) for x in molecules_smiles]
    molecules_smiles_mask = [True] * len(molecules_smiles)
    for i in range(1, len(molecules_smiles)):
        if molecules_smiles_no_stereo[i] is None:
            molecules_smiles_mask[i] = False
        if molecules_smiles_no_stereo[i] in molecules_smiles_no_stereo[:i]:
            molecules_smiles_mask[i] = False
    return [molecules_ids[i] for i in range(len(molecules_ids)) if molecules_smiles_mask[i]]


def load_molecules(file_path):
    supplier = Chem.SDMolSupplier(file_path)
    molecules = [mol for mol in supplier if mol is not None]
    print(f"Loaded {len(molecules)} molecules from {file_path}")
    return molecules


def calculate_average_distance(mols1, mols2):
    if len(mols1) != len(mols2):
        raise ValueError("The number of molecules in the files must be the same.")
    total_distance = 0
    atom_pairs_count = 0
    for mol1, mol2 in zip(mols1, mols2):
        if mol1.GetNumAtoms() != mol2.GetNumAtoms():
            raise ValueError("The molecules must have the same number of atoms.")
        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()
        for atom_idx in range(mol1.GetNumAtoms()):
            pos1 = np.array(conf1.GetAtomPosition(atom_idx))
            pos2 = np.array(conf2.GetAtomPosition(atom_idx))
            distance = np.linalg.norm(pos1 - pos2)
            total_distance += distance
            atom_pairs_count += 1
    average_distance = total_distance / atom_pairs_count if atom_pairs_count > 0 else 0
    return average_distance


if __name__ == "__main__":
    a = "/Users/amitay.s/PycharmProjects/BioCatalystChem/datasets/docking2/A0A009HWM5/47/complex_0/rank10_confidence-4.14.sdf"
    b = "/Users/amitay.s/PycharmProjects/BioCatalystChem/datasets/docking2/A0A009HWM5/47/complex_0/rank1_confidence-1.83.sdf"
    mols1 = load_molecules(a)
    mols2 = load_molecules(b)
    print(calculate_average_distance(mols1, mols2))
