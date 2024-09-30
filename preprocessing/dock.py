import os
import subprocess
import requests
def query_pdb(uniprot_id):
    url = f"https://search.rcsb.org/rcsbsearch/v1/query?json=%7B%22query%22:%7B%22type%22:%22terminal%22,%22service%22:%22text%22,%22parameters%22:%7B%22value%22:%22{uniprot_id}%22,%22attribute%22:%22rcsb_uniprot_container.identifiers.accession%22%7D%7D,%22return_type%22:%22entry%22%7D"
    print(url)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error querying PDB. Status code: {response.status_code}")
    pdb_data = response.json()
    pdb_ids = [entry["identifier"] for entry in pdb_data.get("result_set", [])]
    return pdb_ids

query_pdb("P0A8V2")

3/0



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

aa3to1 = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
    'MSE': 'M',
}


def pdb_to_fasta(pdb_file):
    import re

    # Regular expression to match CA atom lines in the PDB file and capture relevant data
    ca_pattern = re.compile(r"^ATOM\s+\d+\s+CA\s+([A-Z]{3})\s+([\w])\s+\d+\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)")

    # Initialize dictionaries and lists to store chain information and coordinates
    chain_dict = dict()  # Stores the amino acid sequences for each chain
    chain_coords = dict()  # Stores the coordinates of CA atoms for each chain
    chain_list = []

    # Open the PDB file for reading
    with open(pdb_file, 'r') as fp:
        for line in fp.read().splitlines():
            # Break at the end of the model
            if line.startswith("ENDMDL"):
                break

            # Match the pattern for CA atoms
            match = ca_pattern.match(line)
            if match:
                # Extract residue name, chain, and XYZ coordinates
                resn = match.group(1)  # Residue name (e.g., SER)
                chain = match.group(2)  # Chain identifier (e.g., A)
                x_coord = float(match.group(3))  # X coordinate
                y_coord = float(match.group(4))  # Y coordinate
                z_coord = float(match.group(5))  # Z coordinate

                # Get the one-letter code for the residue (you need to define aa3to1 dictionary separately)
                aa1 = aa3to1.get(resn, 'X')  # 'X' as fallback for unknown residues

                # Append the residue to the chain sequence
                if chain in chain_dict:
                    chain_dict[chain] += aa1
                else:
                    chain_dict[chain] = aa1
                    chain_list.append(chain)

                # Store the coordinates
                if chain not in chain_coords:
                    chain_coords[chain] = []
                chain_coords[chain].append((x_coord, y_coord, z_coord))

    # Return the sequences and coordinates
    return chain_dict, chain_list, chain_coords


def extract_3d_coordinates_from_mol2(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    in_atom_section = False

    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            continue
        elif line.startswith("@<TRIPOS>"):
            in_atom_section = False

        # If in the ATOM section, extract the 3D coordinates
        if in_atom_section:
            parts = line.split()
            if len(parts) >= 5:
                atom_id = parts[0]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                coordinates.append((atom_id, x, y, z))

    return coordinates


# Load the PDB file
protein_file = '/Users/amitay.s/PycharmProjects/BioCatalystChem/4agq/4agq_protein.pdb'

chain_dict, chain_list, chain_coords = pdb_to_fasta(protein_file)
fasta=chain_dict[chain_list[0]]
print(fasta)
chain_coords = chain_coords[chain_list[0]]
chain_coords = np.array(chain_coords)
# Load the mol2 ligand file
ligand_file = '/Users/amitay.s/PycharmProjects/BioCatalystChem/4agq/4agq_ligand.mol2'
with open(ligand_file, 'r') as f:
    ligand_data = f.read()

from rdkit import Chem

# Load the .mol2 file
mol = Chem.MolFromMol2File(ligand_file, removeHs=False)

# Convert to SMILES
if mol:
    smiles = Chem.MolToSmiles(mol)
    print("SMILES:", smiles)
else:
    print("Failed to load the molecule")


lig_coords = extract_3d_coordinates_from_mol2(ligand_file)
ligand_locs = np.array([x[1:] for x in lig_coords])
dist = euclidean_distances(chain_coords, ligand_locs)

# dist = dist.min(axis=1)
weights = np.exp(-dist)
weights = weights / weights.sum(axis=0)
weights = weights.sum(axis=1)
weights=weights/weights.sum()


# sotred_dist = np.sort(dist)
fig, ax = plt.subplots()
ax.plot(weights, 'b')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(chain_coords[:, 0], chain_coords[:, 1], chain_coords[:, 2], c=weights, marker='o', cmap='RdYlGn')
ax.scatter(ligand_locs[:, 0], ligand_locs[:, 1], ligand_locs[:, 2], c='b', marker='o')

plt.show()
