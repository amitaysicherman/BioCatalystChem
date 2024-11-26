import os
from vis.utils import load_maps, remove_stereo, remove_dup_mis_mols

id_to_smile, smile_to_id, uniport_to_ec = load_maps()

base_dir = "datasets/docking2"
for prot in os.listdir(base_dir):
    if not os.path.isdir(os.path.join(base_dir, prot)):
        continue
    if len(os.listdir(os.path.join(base_dir, prot))) == 0:
        continue
    mols = os.listdir(os.path.join(base_dir, prot))
    mols = remove_dup_mis_mols(mols, id_to_smile)
    print(f"Found {len(mols)} unique molecules for protein {prot}")

