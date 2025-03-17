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

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

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

def plot_learning_curves(train_losses, val_losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.savefig(f'data/{filename}')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load full GDB17 data
    gdb17_data = pd.read_csv('data/GDB17.csv')
    gdb17_smiles = gdb17_data['SMILES'].tolist()
    
    # Select 400,000 SMILES (shuffle first to avoid bias)
    random.shuffle(gdb17_smiles)
    selected_gdb17_smiles = gdb17_smiles[:400000]
    valid_gdb17 = fetch_valid_smiles(selected_gdb17_smiles)
    logging.info(f"Loaded {len(valid_gdb17)} valid SMILES from GDB17")
    
    # Preprocessing
    combined_smiles = ['!' + smi for smi in valid_gdb17]
    start_chars = [smi[0] for smi in combined_smiles if smi]
    start_char_counts = {c: start_chars.count(c) for c in set(start_chars)}
    start_char_weights = [count/len(start_chars) for count in start_char_counts.values()]
    
    chars = sorted(list(set(''.join(combined_smiles))) + ['<PAD>', '<EOS>'])
    char_to_idx = {c:i for i,c in enumerate(chars)}
    idx_to_char = {i:c for i,c in enumerate(chars)}
    
    # Dynamic sequence length calculation
    seq_lengths = [len(smi) for smi in combined_smiles]
    seq_len = int(np.percentile(seq_lengths, 95)) + 1
    logging.info(f"Using sequence length: {seq_len}")
    
    # Reduced model parameters for VRAM efficiency
    model = CLM(
        vocab_size=len(chars),
        embed_dim=64,
        hidden_dim=128,
        n_layers=2,
        dropout=0.1
    ).to(device)
    
    # Pretraining with optimizations
    logging.info("Starting pre-training on 400K SMILES")
    train_losses, val_losses = pretrain_clm(
        model,
        combined_smiles,
        char_to_idx,
        device,
        epochs=15,
        batch_size=32,  # Reduced for memory
        lr=0.0005,
        seq_len=seq_len
    )
    plot_learning_curves(train_losses, val_losses, 
                        'Pretraining (400K SMILES)', 'pretraining_400k.png')
    
    # Fine-tuning setup
    small_data = pd.read_csv('data/SMILE.csv')
    small_smiles = ['!' + smi for smi in fetch_valid_smiles(small_data['SMILES'].tolist())]
    train_set, val_set = train_test_split(small_smiles, test_size=0.2, random_state=42)
    
    # Fine-tuning with lower batch size and learning rate scheduling
    logging.info("Starting fine-tuning")
    train_losses, val_losses = finetune_clm(
        model,
        train_set,
        char_to_idx,
        device,
        epochs=20,
        batch_size=16,
        lr=0.0001,
        seq_len=seq_len
    )
    plot_learning_curves(train_losses, val_losses, 
                        'Fine-tuning', 'finetuning_400k.png')
    
    # Evaluation and generation
    dataset = SMILESDataset(train_set, char_to_idx, seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    eval_loss = evaluate_model(model, loader, device)
    logging.info(f"Evaluation Loss: {eval_loss:.4f}")
    
    # Generate new SMILES
    generated = []
    for _ in range(50):
        start_chars = random.choices(list(start_char_counts.keys()), 
                                    weights=start_char_weights, 
                                    k=32)
        batch = generate_smiles_batch(model, char_to_idx, idx_to_char, start_chars, seq_len)
        generated += [s[1:] for s in batch if Chem.MolFromSmiles(s[1:])]
    
    pd.DataFrame(generated, columns=['SMILES']).to_csv('data/generated_smiles_400k.csv', index=False)

if __name__ == "__main__":
    main()