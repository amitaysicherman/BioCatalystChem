from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration, T5Config
from preprocessing.build_tokenizer import get_tokenizer_file_path
from model import CustomT5Model
import torch
from dataset_v2 import SeqToSeqDataset
from finetune_ecreact_v2 import CustomDataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokens_to_canonical_smiles(tokenizer, tokens, remove_stereo=True):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def load_model(cp, custom_model, daa_type=3, add_mode=0):
    if not custom_model:
        model = T5ForConditionalGeneration.from_pretrained(cp)
    else:
        print("Loading custom model", cp)
        config = T5Config.from_pretrained(cp)
        model = CustomT5Model(config, daa_type=daa_type, add_mode=add_mode)
        model.load_state_dict(torch.load(f"{cp}/pytorch_model.bin"))
    model.to(device)
    model.eval()
    return model


def load_tokenizer():
    return PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())





baseline_cp = "results/NO_EC_nmix/checkpoint-12730"
baseline_model = load_model(baseline_cp, custom_model=False)
model_cp = "results/PRETRAINED_3_nmix/checkpoint-12730"
model = load_model(model_cp, custom_model=True)
tokenizer = load_tokenizer()
dataset_args = {"datasets": ["ecreact/level4"], "split": "test", "tokenizer": tokenizer}
dataset = DataLoader(SeqToSeqDataset(**dataset_args, add_emb=[True]), batch_size=1, num_workers=0,
                     collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, model=model))

def batch_to_type(batch):
    batch_ids = batch['id'].detach().cpu().numpy().flatten().tolist()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device).bool()
    labels = batch['labels'].to(device)
    emb = batch['emb'].to(device).float()
    scores = batch['docking_scores'].to(device).float()
    emb_mask = batch['emb_mask'].to(device).bool()
    return batch_ids, input_ids, attention_mask, labels, emb, scores, emb_mask



import matplotlib.pyplot as plt
from rdkit.Chem import Draw


def get_difference_atoms(correct_mol, incorrect_mol):
    """
    Identifies the atoms that differ between two molecules.

    Args:
        correct_mol: RDKit Mol object for the correct prediction.
        incorrect_mol: RDKit Mol object for the incorrect prediction.

    Returns:
        List of atom indices in the incorrect molecule that differ.
    """
    # Ensure both molecules are valid
    if correct_mol is None or incorrect_mol is None:
        return list(range(incorrect_mol.GetNumAtoms() if incorrect_mol else 0))

    # Perform atom-by-atom comparison
    differing_atoms = []
    for idx in range(min(correct_mol.GetNumAtoms(), incorrect_mol.GetNumAtoms())):
        correct_atom = correct_mol.GetAtomWithIdx(idx)
        incorrect_atom = incorrect_mol.GetAtomWithIdx(idx)
        if correct_atom.GetSymbol() != incorrect_atom.GetSymbol():
            differing_atoms.append(idx)

    # Include additional atoms if molecule sizes differ
    if correct_mol.GetNumAtoms() != incorrect_mol.GetNumAtoms():
        differing_atoms.extend(range(min(correct_mol.GetNumAtoms(), incorrect_mol.GetNumAtoms()), incorrect_mol.GetNumAtoms()))

    return differing_atoms
def get_difference_bonds(correct_mol, incorrect_mol):
    """
    Identifies the bonds that differ between two molecules.

    Args:
        correct_mol: RDKit Mol object for the correct prediction.
        incorrect_mol: RDKit Mol object for the incorrect prediction.

    Returns:
        List of bond indices in the incorrect molecule that differ.
    """
    differing_bonds = []
    for bond_idx in range(min(correct_mol.GetNumBonds(), incorrect_mol.GetNumBonds())):
        correct_bond = correct_mol.GetBondWithIdx(bond_idx)
        incorrect_bond = incorrect_mol.GetBondWithIdx(bond_idx)
        if correct_bond.GetBeginAtomIdx() != incorrect_bond.GetBeginAtomIdx() or \
           correct_bond.GetEndAtomIdx() != incorrect_bond.GetEndAtomIdx() or \
           correct_bond.GetBondType() != incorrect_bond.GetBondType():
            differing_bonds.append(bond_idx)

    return differing_bonds


def plot_difference_with_annotations(correct_smiles, incorrect_smiles, label_smiles, title):
    """
    Plots molecules and highlights differences in atoms or bonds.

    Args:
        correct_smiles: SMILES string of the correct prediction.
        incorrect_smiles: SMILES string of the incorrect prediction.
        label_smiles: SMILES string of the ground truth label.
        title: Title for the plot.
    """
    correct_mol = Chem.MolFromSmiles(correct_smiles)
    incorrect_mol = Chem.MolFromSmiles(incorrect_smiles)
    label_mol = Chem.MolFromSmiles(label_smiles)

    if correct_mol is None or incorrect_mol is None or label_mol is None:
        print("Invalid SMILES for plotting.")
        return

    # Get differing atoms and bonds
    differing_atoms = get_difference_atoms(correct_mol, incorrect_mol)
    differing_bonds = get_difference_bonds(correct_mol, incorrect_mol)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].set_title("Ground Truth")
    axes[1].set_title("Correct Prediction")
    axes[2].set_title("Incorrect Prediction")

    # Plot ground truth
    img_label = Draw.MolToImage(label_mol, size=(300, 300), kekulize=True, fitImage=True)
    axes[0].imshow(img_label)
    axes[0].axis("off")

    # Plot correct prediction
    img_correct = Draw.MolToImage(
        correct_mol, highlightAtoms=[], highlightBonds=[], size=(300, 300)
    )
    axes[1].imshow(img_correct)
    axes[1].axis("off")

    # Plot incorrect prediction with highlighted atoms and bonds
    img_incorrect = Draw.MolToImage(
        incorrect_mol,
        highlightAtoms=differing_atoms,
        highlightBonds=differing_bonds,
        size=(300, 300),
    )
    axes[2].imshow(img_incorrect)
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


for i, batch in tqdm(enumerate(dataset), total=len(dataset)):
    batch_ids, input_ids, attention_mask, labels, emb, scores, emb_mask = batch_to_type(batch)

    emb_args = {"emb": emb, "emb_mask": emb_mask, "docking_scores": scores}
    args = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    outputs = model(**args, **emb_args).logits.argmax(dim=-1)[0]
    baseline_outputs = baseline_model(**args).logits.argmax(dim=-1)[0]
    labels = labels[0]
    label = labels[(labels != tokenizer.pad_token_id) & (labels != -100)]
    label_smiles = tokens_to_canonical_smiles(tokenizer, label)
    model_preds = tokens_to_canonical_smiles(tokenizer, outputs)
    baseline_preds = tokens_to_canonical_smiles(tokenizer, baseline_outputs)
    id_ = batch_ids[0]
    model_is_correct = label_smiles == model_preds
    baseline_is_correct = label_smiles == baseline_preds
    if model_is_correct and not baseline_is_correct:
        print(f"Found first case where model is correct and baseline is not, at batch ID {id_}.")
        plot_difference_with_annotations(
            correct_smiles=model_preds,
            incorrect_smiles=baseline_preds,
            label_smiles=label_smiles,
            title=f"Batch ID: {id_}",
        )

