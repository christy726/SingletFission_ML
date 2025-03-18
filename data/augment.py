import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import AllChem as Chem
import random
import time

def load_smiles_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['SMILES'].tolist()

def get_rand_smi(smi, max_folds=10):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    smi_list = set()
    for _ in range(max_folds):
        s = Chem.MolToSmiles(mol, doRandom=True)
        smi_list.add(s)
    smi_list = list(smi_list)
    return smi_list

def has_valid_atom_count(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    num_atoms = mol.GetNumAtoms()
    return 15 <= num_atoms <= 50

def augmentation_by_smi(lst_smi, max_folds=10):
    if max_folds <= 1:
        return lst_smi

    list_of_augmented_smi = []
    for smi in lst_smi:
        list_of_rand_smi = get_rand_smi(smi, max_folds)
        valid_smis = [s for s in list_of_rand_smi if has_valid_atom_count(s)]
        list_of_augmented_smi.extend(valid_smis)

    return list_of_augmented_smi

def replace_number(smi: str, n=1) -> str:
    s = ''
    for char in smi:
        new_char = str(int(char) + n) if char.isdigit() else char
        s += new_char
    return s

def combine_fragments(fragments, n_combs: int):
    list_of_smi = []
    singles = [frag for frag in fragments if frag.count('*') == 1]
    n_frags, n_singles = len(fragments), len(singles)
    print(f'{n_frags} {n_singles}')
    for _ in range(n_combs):
        idx = random.randint(0, n_frags - 1)
        bone = fragments[idx]
        bone = replace_number(bone)
        for i in range(bone.count('*')):
            frag = singles[random.randint(0, n_singles - 1)]
            frag = frag.replace('*', '')
            bone = bone.replace('*', frag, 1)
        bone = bone.replace('*', '')
        try:
            mol = Chem.MolFromSmiles(bone)
            Chem.SanitizeMol(mol)
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            if has_valid_atom_count(canonical_smi):
                list_of_smi.append(canonical_smi)
        except:
            pass
    return list_of_smi

def augmentation_by_fragment(list_of_smi, n: int):
    fragments = set()
    for smi in list_of_smi:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        hierarch = Recap.RecapDecompose(m)
        leaves = list(hierarch.GetLeaves().keys())
        fragments.update(leaves)

    fragments = list(fragments)
    print('get {} fragments'.format(len(fragments)))
    return combine_fragments(fragments, n)

def generate_augmented_smiles(file_path, target_count=200000, max_folds=10, fragment_multiplier=10):
    original_smis = load_smiles_from_csv(file_path)
    print(f'Loaded {len(original_smis)} original SMILES.')

    augmented_smis = augmentation_by_smi(original_smis, max_folds=max_folds)
    print(f'Generated {len(augmented_smis)} SMILES by random variations.')

    fragment_smis = augmentation_by_fragment(original_smis, n=target_count // fragment_multiplier)
    print(f'Generated {len(fragment_smis)} SMILES by fragment-based combinations.')

    all_augmented_smis = augmented_smis + fragment_smis
    print(f'Total augmented SMILES: {len(all_augmented_smis)}')

    unique_augmented_smis = list(set(all_augmented_smis))
    final_augmented_smis = unique_augmented_smis[:target_count]
    print(f'Final augmented SMILES count after deduplication: {len(final_augmented_smis)}')

    return final_augmented_smis

if __name__ == "__main__":
    start_time = time.time()
    augmented_smis = generate_augmented_smiles('SMILE.csv', target_count=20000000, max_folds=10, fragment_multiplier=10)
    end_time = time.time()
    print(f'Process completed in {end_time - start_time:.2f} seconds.')

    output_df = pd.DataFrame({'SMILES': augmented_smis})
    output_df.to_csv('augmented_SMILE_18_03_2025.csv', index=False)
    print('Augmented SMILES saved to augmented_SMILE.csv')