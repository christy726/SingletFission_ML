import pandas as pd
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Recap, AllChem as Chem
import random
from tqdm import tqdm
import sys

INPUT_FILE = r"D:\My_Computer\Meh!!!!!!!\workspace\VIT_LAB\SingletFission_ML\data\SMILE.csv"
OUTPUT_FILE = r"D:\My_Computer\Meh!!!!!!!\workspace\VIT_LAB\SingletFission_ML\data\augmented_smiles.csv"
TOTAL_TARGET = 600000  # Reduce to 10000 for testing
MAX_FOLDS = 8
FRAGMENT_BATCH_SIZE = 200

def get_rand_smi(smi, max_folds=MAX_FOLDS):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return []
    variants = set()
    for _ in range(max_folds):
        try:
            variant = Chem.MolToSmiles(mol, doRandom=True)
            variants.add(variant)
        except Exception as e:
            print(f"Error generating variant for {smi}: {e}")
    return list(variants)

def fragment_decomposition(smi_list):
    fragments = set()
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                hierarch = Recap.RecapDecompose(mol)
                leaves = list(hierarch.GetLeaves().keys())
                fragments.update(leaves)
            except Exception as e:
                print(f"Error decomposing {smi}: {e}")
    return list(fragments)

def combine_fragments(fragments, n):
    singles = [f for f in fragments if f.count('*') == 1]
    results = []
    for _ in range(n):
        try:
            bone = random.choice(fragments)
            bone = replace_number(bone, random.randint(1, 3))
            for _ in range(bone.count('*')):
                if not singles:
                    break
                frag = random.choice(singles).replace('*', '')
                bone = bone.replace('*', frag, 1)
            bone = bone.replace('*', '')
            mol = Chem.MolFromSmiles(bone)
            if mol:
                results.append(Chem.MolToSmiles(mol))
        except Exception as e:
            print(f"Fragment combination failed: {e}")
    return results

def replace_number(smi, n=1):
    return ''.join([str(int(c)+n) if c.isdigit() else c for c in smi])

def process_batch(smi_batch):
    return [get_rand_smi(smi) for smi in smi_batch]

def main():
    try:
        # Load data
        df = pd.read_csv(INPUT_FILE)
        smis = df["SMILES"].tolist()
        print(f"Loaded {len(smis)} original SMILES")

        # Validate input SMILES
        valid_smis = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_smis.append(smi)
        print(f"Valid SMILES after filtering: {len(valid_smis)}")
        smis = valid_smis

        # Step 1: Random SMILES generation (parallel)
        with multiprocessing.Pool(processes=4) as pool:
            batches = [smis[i::4] for i in range(4)]
            results = pool.map(process_batch, batches)
        augmented = []
        for res in results:
            for variants in res:
                augmented.extend(variants)
        augmented = list(set(augmented))
        print(f"Step 1: {len(augmented)} variants generated")

        # Step 2: Fragment-based augmentation
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

        # Deduplicate
        augmented = list(set(augmented))
        print(f"Final count: {len(augmented)} SMILES (deduplicated)")

        # Save to CSV (force save even if target not met)
        if len(augmented) < 1:
            print("No valid SMILES generated!")
        else:
            # Save in chunks to avoid memory issues
            chunk_size = 10000
            for i in range(0, len(augmented), chunk_size):
                chunk = augmented[i:i+chunk_size]
                df_chunk = pd.DataFrame({"SMILES": chunk})
                df_chunk.to_csv(
                    f"{OUTPUT_FILE[:-4]}_{i//chunk_size}.csv",
                    index=False,
                    mode='a' if i > 0 else 'w',
                    header=(i == 0)
                )
            print(f"Saved {len(augmented)} SMILES to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()