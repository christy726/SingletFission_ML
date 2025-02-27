import torch
import torch.optim as optim
from tqdm import tqdm
from utils.dataset import SMILESDatasetOneHot
import logging

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc='Training', leave=False)
    for batch in progress_bar:
        inputs = batch[:, :-1].to(device)   
        targets = batch[:, 1:].contiguous().view(-1).to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs, None)

        outputs = outputs.reshape(-1, outputs.size(-1))

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})

    avg_loss = total_loss / len(loader)
    return avg_loss

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', leave=False):
            inputs = batch[:, :-1].to(device)  
            targets = batch[:, 1:].contiguous().view(-1).to(device)
            outputs, _ = model(inputs, None)
            outputs = outputs.reshape(-1, outputs.size(-1))

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def pretrain_clm(model, smiles, char_to_idx, device, epochs=30, batch_size=128, lr=0.001, patience=5):
    dataset = SMILESDatasetOneHot(smiles, char_to_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        avg_train_loss = train_epoch(model, loader, criterion, optimizer, device)
        avg_val_loss = validate_epoch(model, loader, criterion, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        logging.info(f'Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_pretrained_model.pt')
            logging.info("Model saved to 'models/best_pretrained_model.pt'")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break

    return train_losses, val_losses

def finetune_clm(model, smiles, char_to_idx, device, epochs=10, batch_size=32, lr=0.0001, patience=5):
    dataset = SMILESDatasetOneHot(smiles, char_to_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        avg_train_loss = train_epoch(model, loader, criterion, optimizer, device)
        avg_val_loss = validate_epoch(model, loader, criterion, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        logging.info(f'Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_finetuned_model.pt')
            logging.info("Model saved to 'models/best_finetuned_model.pt'")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break

    return train_losses, val_losses