
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.dataset import SMILESDataset
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

def pretrain_clm(model, smiles, char_to_idx, device, epochs=30, batch_size=128, lr=0.001, seq_len=100):
    dataset = SMILESDataset(smiles, char_to_idx, seq_len=seq_len)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch in progress_bar:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        model.train()
        logging.info(f'Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {val_loss:.4f}')
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/pretrained_clm.pt')
            logging.info("Best model saved to 'models/pretrained_clm.pt'")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
    return train_losses, val_losses

def finetune_clm(model, smiles, char_to_idx, device, epochs=10, batch_size=32, lr=0.0001, seq_len=100):
    dataset = SMILESDataset(smiles, char_to_idx, seq_len=seq_len)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch in progress_bar:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        model.train()
        logging.info(f'Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {val_loss:.4f}')
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/finetuned_clm.pt')
            logging.info("Best model saved to 'models/finetuned_clm.pt'")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
    return train_losses, val_losses