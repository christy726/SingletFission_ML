# import pandas as pd
# import multiprocessing
# from rdkit import Chem
# from rdkit.Chem import Recap, AllChem as Chem
# import random
# from tqdm import tqdm
# import sys

# INPUT_FILE = r"D:\My_Computer\Meh!!!!!!!\workspace\VIT_LAB\SingletFission_ML\data\SMILE.csv"
# OUTPUT_FILE = r"D:\My_Computer\Meh!!!!!!!\workspace\VIT_LAB\SingletFission_ML\data\augmented_smiles.csv"
# TOTAL_TARGET = 600000  # Reduce to 10000 for testing
# MAX_FOLDS = 8
# FRAGMENT_BATCH_SIZE = 200

# def get_rand_smi(smi, max_folds=MAX_FOLDS):
#     mol = Chem.MolFromSmiles(smi)
#     if not mol:
#         return []
#     variants = set()
#     for _ in range(max_folds):
#         try:
#             variant = Chem.MolToSmiles(mol, doRandom=True)
#             variants.add(variant)
#         except Exception as e:
#             print(f"Error generating variant for {smi}: {e}")
#     return list(variants)

# def fragment_decomposition(smi_list):
#     fragments = set()
#     for smi in smi_list:
#         mol = Chem.MolFromSmiles(smi)
#         if mol:
#             try:
#                 hierarch = Recap.RecapDecompose(mol)
#                 leaves = list(hierarch.GetLeaves().keys())
#                 fragments.update(leaves)
#             except Exception as e:
#                 print(f"Error decomposing {smi}: {e}")
#     return list(fragments)

# def combine_fragments(fragments, n):
#     singles = [f for f in fragments if f.count('*') == 1]
#     results = []
#     for _ in range(n):
#         try:
#             bone = random.choice(fragments)
#             bone = replace_number(bone, random.randint(1, 3))
#             for _ in range(bone.count('*')):
#                 if not singles:
#                     break
#                 frag = random.choice(singles).replace('*', '')
#                 bone = bone.replace('*', frag, 1)
#             bone = bone.replace('*', '')
#             mol = Chem.MolFromSmiles(bone)
#             if mol:
#                 results.append(Chem.MolToSmiles(mol))
#         except Exception as e:
#             print(f"Fragment combination failed: {e}")
#     return results

# def replace_number(smi, n=1):
#     return ''.join([str(int(c)+n) if c.isdigit() else c for c in smi])

# def process_batch(smi_batch):
#     return [get_rand_smi(smi) for smi in smi_batch]

# def main():
#     try:
#         # Load data
#         df = pd.read_csv(INPUT_FILE)
#         smis = df["SMILES"].tolist()
#         print(f"Loaded {len(smis)} original SMILES")

#         # Validate input SMILES
#         valid_smis = []
#         for smi in smis:
#             mol = Chem.MolFromSmiles(smi)
#             if mol:
#                 valid_smis.append(smi)
#         print(f"Valid SMILES after filtering: {len(valid_smis)}")
#         smis = valid_smis

#         # Step 1: Random SMILES generation (parallel)
#         with multiprocessing.Pool(processes=4) as pool:
#             batches = [smis[i::4] for i in range(4)]
#             results = pool.map(process_batch, batches)
#         augmented = []
#         for res in results:
#             for variants in res:
#                 augmented.extend(variants)
#         augmented = list(set(augmented))
#         print(f"Step 1: {len(augmented)} variants generated")

#         # Step 2: Fragment-based augmentation
#         fragments = fragment_decomposition(augmented)
#         print(f"Extracted {len(fragments)} unique fragments")

#         current_count = len(augmented)
#         pbar = tqdm(total=TOTAL_TARGET)
#         while current_count < TOTAL_TARGET:
#             batch_size = min(FRAGMENT_BATCH_SIZE, TOTAL_TARGET - current_count)
#             new_smis = combine_fragments(fragments, batch_size)
#             augmented += new_smis
#             current_count = len(augmented)
#             pbar.update(len(new_smis))
#         pbar.close()

#         # Deduplicate
#         augmented = list(set(augmented))
#         print(f"Final count: {len(augmented)} SMILES (deduplicated)")

#         # Save to CSV (force save even if target not met)
#         if len(augmented) < 1:
#             print("No valid SMILES generated!")
#         else:
#             # Save in chunks to avoid memory issues
#             chunk_size = 10000
#             for i in range(0, len(augmented), chunk_size):
#                 chunk = augmented[i:i+chunk_size]
#                 df_chunk = pd.DataFrame({"SMILES": chunk})
#                 df_chunk.to_csv(
#                     f"{OUTPUT_FILE[:-4]}_{i//chunk_size}.csv",
#                     index=False,
#                     mode='a' if i > 0 else 'w',
#                     header=(i == 0)
#                 )
#             print(f"Saved {len(augmented)} SMILES to {OUTPUT_FILE}")

#     except Exception as e:
#         print(f"Fatal error: {e}")

# if __name__ == "__main__":
#     main()


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
    augmented_smis = generate_augmented_smiles('SMILE.csv', target_count=50000000, max_folds=10, fragment_multiplier=10)
    end_time = time.time()
    print(f'Process completed in {end_time - start_time:.2f} seconds.')

    output_df = pd.DataFrame({'SMILES': augmented_smis})
    output_df.to_csv('augmented_SMILE.csv', index=False)
    print('Augmented SMILES saved to augmented_SMILE.csv')