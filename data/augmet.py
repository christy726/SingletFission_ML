import pandas as pd
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Recap, AllChem as Chem
import random
from tqdm import tqdm
import sys

INPUT_FILE = "D:\My_Computer\Meh!!!!!!!\workspace\VIT_LAB\SingletFission_ML\data\SMILE.csv"
OUTPUT_FILE = "D:\My_Computer\Meh!!!!!!!\workspace\VIT_LAB\SingletFission_ML\data\augmented_smiles.csv"
TOTAL_TARGET = 600000
MAX_FOLDS = 8   
FRAGMENT_BATCH_SIZE = 200   

def get_rand_smi(smi, max_folds=MAX_FOLDS):
    """Generate random SMILES variants for a single molecule."""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return []
    variants = set()
    for _ in range(max_folds):
        try:
            variant = Chem.MolToSmiles(mol, doRandom=True)
            variants.add(variant)
        except:
            continue
    return list(variants)

def fragment_decomposition(smi_list):
    """Extract fragments from SMILES using Recap."""
    fragments = set()
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            hierarch = Recap.RecapDecompose(mol)
            leaves = list(hierarch.GetLeaves().keys())
            fragments.update(leaves)
    return list(fragments)

def combine_fragments(fragments, n):
    """Generate new molecules by combining fragments."""
    singles = [f for f in fragments if f.count('*') == 1]
    results = []
    for _ in range(n):
        bone = random.choice(fragments)
        bone = replace_number(bone, random.randint(1, 3))
        for _ in range(bone.count('*')):
            if not singles:
                break
            frag = random.choice(singles).replace('*', '')
            bone = bone.replace('*', frag, 1)
        bone = bone.replace('*', '')
        try:
            mol = Chem.MolFromSmiles(bone)
            if mol:
                results.append(Chem.MolToSmiles(mol))
        except:
            pass
    return results

def replace_number(smi, n=1):
    """Shift numeric labels for fragment compatibility."""
    return ''.join([str(int(c)+n) if c.isdigit() else c for c in smi])

def process_batch(smi_batch):
    """Parallel processing for random SMILES generation."""
    return [get_rand_smi(smi) for smi in smi_batch]

def main():
    df = pd.read_csv(INPUT_FILE)
    smis = df["SMILES"].tolist()
    print(f"Loaded {len(smis)} original SMILES")

    with multiprocessing.Pool(processes=4) as pool:
        batches = [smis[i::4] for i in range(4)]
        results = pool.map(process_batch, batches)
    augmented = []
    for res in results:
        for variants in res:
            augmented.extend(variants)
    augmented = list(set(augmented))
    print(f"Step 1: {len(augmented)} variants generated")

    fragments = fragment_decomposition(augmented)
    print(f"Extracted {len(fragments)} unique fragments")

    current_count = len(augmented)
    pbar = tqdm(total=TOTAL_TARGET)
    while current_count < TOTAL_TARGET:
        batch_size = min(FRAGMENT_BATCH_SIZE, TOTAL_TARGET - current_count)
        new_smis = combine_fragments(fragments, batch_size)
        augmented += new_smis
        current_count = len(augmented)
        pbar.update(len(new_smis))
    pbar.close()

    augmented = list(set(augmented))
    print(f"Final count: {len(augmented)} SMILES (deduplicated)")

    if len(augmented) < TOTAL_TARGET:
        print("Warning: Could not reach target. Adjust parameters.")
    else:
        df_out = pd.DataFrame({"SMILES": augmented[:TOTAL_TARGET]})
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()