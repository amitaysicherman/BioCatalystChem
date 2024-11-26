import os
from vis.utils import load_maps, remove_dup_mis_mols, filter_molecule_by_len
import glob

id_to_smile, smile_to_id, uniport_to_ec = load_maps()

base_dir = "datasets/docking2"
for protein_id in os.listdir(base_dir):
    protein_ec = uniport_to_ec[protein_id]
    molecules_ids = os.listdir(f"datasets/docking2/{protein_id}")
    molecules_ids = remove_dup_mis_mols(molecules_ids, id_to_smile)
    m_dock_positions = []
    for m in molecules_ids:
        sdf_files = glob.glob(f"datasets/docking2/{protein_id}/{m}/complex_0/*.sdf")
        try:
            sdf_files = filter_molecule_by_len(sdf_files, 0.5)
            m_dock_positions.append(len(sdf_files))
        except Exception as e:
            print(f"Error in {protein_id} - {m}", e)
    if len(molecules_ids) > 5 and len(molecules_ids) < 10 and sum([x > 2 for x in m_dock_positions]) > 3:
        print(f"Protein {protein_id} - {len(molecules_ids)} molecules, {m_dock_positions}")
