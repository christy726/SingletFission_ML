import torch
import logging
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from utils.dataset import SMILESDataset
from models.clm_model import CLM
from utils.train_utils import train_model, generate_smiles_batch, evaluate_model
from utils.clm_utils import pretrain_clm, finetune_clm
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_valid_smiles(smiles_list):
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_smiles.append(Chem.MolToSmiles(mol))
        except Exception as e:
            logging.warning(f"Invalid SMILES string skipped: {smi}, error: {e}")
    return valid_smiles

def load_augmented_data(filepath):
    try:
        augmented_data = pd.read_csv(filepath)
        logging.info("Augmented dataset loaded")
        return augmented_data['SMILES'].tolist()
    except FileNotFoundError:
        logging.error("Augmented dataset not found. Please check the file path.")
        return []

def plot_learning_curves(train_losses, val_losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.savefig(f'data/{filename}')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load and preprocess data
    gdb17_data = pd.read_csv('data/GDB17.csv')
    gdb17_smiles = gdb17_data['SMILES'].tolist()
    random.shuffle(gdb17_smiles)
    selected_gdb17_smiles = gdb17_smiles[:300000]
    valid_gdb17 = fetch_valid_smiles(selected_gdb17_smiles)
    logging.info(f"Collected {len(valid_gdb17)} valid SMILES from GDB17")

    augmented_smiles = load_augmented_data('data/augmented.csv')
    valid_augmented = fetch_valid_smiles(augmented_smiles)
    logging.info(f"Collected {len(valid_augmented)} valid SMILES from augmented dataset")

    combined_smiles = valid_gdb17 + valid_augmented
    logging.info(f"Combined {len(combined_smiles)} valid SMILES")

    start_chars = [smi[0] for smi in combined_smiles if len(smi) > 0]
    start_char_counts = {}
    for c in start_chars:
        start_char_counts[c] = start_char_counts.get(c, 0) + 1
    total_start = len(start_chars)
    start_char_list = list(start_char_counts.keys())
    start_char_weights = [count/total_start for count in start_char_counts.values()]

    chars = sorted(list(set(''.join(combined_smiles)))) + ['<PAD>', '<EOS>']
    logging.info(f"Vocabulary size: {len(chars)}")
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    seq_lengths = [len(smi) for smi in combined_smiles]
    seq_len = int(np.percentile(seq_lengths, 95)) + 1  
    logging.info(f"Determined sequence length: {seq_len}")

    logging.info("Pre-training CLM on combined dataset...")
    clm_model = CLM(vocab_size=len(chars),embed_dim=128,hidden_dim=256,n_layers=3,dropout=0.2).to(device)
    train_losses, val_losses = pretrain_clm(clm_model,combined_smiles,char_to_idx,device,epochs=5,batch_size=128,lr=0.001,seq_len=seq_len)
    
    logging.info("CLM pre-training complete")
    plot_learning_curves(train_losses, val_losses, 'Pre-training Learning Curves', 'pretraining_learning_curves.png')

    logging.info("Loading smaller dataset for fine-tuning...")
    small_dataset = pd.read_csv('data/SMILE.csv')
    small_smiles = small_dataset['SMILES'].tolist()
    valid_small_smiles = fetch_valid_smiles(small_smiles)
    logging.info(f"Loaded {len(valid_small_smiles)} valid SMILES from smaller dataset")

    train_smiles, val_smiles = train_test_split(valid_small_smiles, test_size=0.1, random_state=42)
    logging.info(f"Train set size: {len(train_smiles)}, Validation set size: {len(val_smiles)}")

    logging.info("Fine-tuning CLM on smaller dataset...")
    train_losses, val_losses = finetune_clm(clm_model,train_smiles,char_to_idx,device,epochs=15,batch_size=32,lr=0.0001,seq_len=seq_len)
    logging.info("CLM fine-tuning complete")
    plot_learning_curves(train_losses, val_losses, 'Fine-tuning Learning Curves', 'finetuning_learning_curves.png')
    
    logging.info("Evaluating CLM...")
    dataset = SMILESDataset(train_smiles, char_to_idx, seq_len=seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    evaluate_model(clm_model, loader, criterion, device)
    logging.info("Evaluation complete")

    logging.info("Generating new SMILES...")
    batch_size = 1000  
    num_batches = 80  
    new_smiles = []
    for _ in tqdm(range(num_batches), desc="Generating SMILES"):
        start_chars = random.choices(start_char_list, weights=start_char_weights, k=batch_size)
        batch_smiles = generate_smiles_batch(clm_model, char_to_idx, idx_to_char, start_chars, max_length=seq_len)
        new_smiles.extend(batch_smiles)

    # Save and validate results
    pd.DataFrame(new_smiles, columns=['SMILES']).to_csv('data/generated_smiles_11_03_2025(2).csv', index=False)
    logging.info("Generated SMILES saved to 'data/generated_smiles_11_03_2025(2).csv'")
    logging.info("Filtering valid and unique SMILES...")
    valid_new_smiles = []
    seen = set(valid_small_smiles)
    for smi in new_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol and smi not in seen:
            seen.add(smi)
            valid_new_smiles.append(smi)
    logging.info(f"Generated {len(valid_new_smiles)} valid and unique SMILES")
    pd.DataFrame(valid_new_smiles, columns=['SMILES']).to_csv('data/valid_unique_smiles_11_03_2025(2).csv', index=False)
    logging.info("Valid and unique SMILES saved to 'data/valid_unique_smiles_11_03_2025(2).csv'")

if __name__ == "__main__":
    main()