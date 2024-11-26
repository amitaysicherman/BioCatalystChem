from Bio.PDB import PDBParser
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
