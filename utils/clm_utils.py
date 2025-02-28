import torch
import torch.optim as optim
from tqdm import tqdm
from utils.dataset import SMILESDataset
import logging

def pretrain_clm(model, smiles, char_to_idx, device, epochs=30, batch_size=128, lr=0.001):
    dataset = SMILESDataset(smiles, char_to_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch in progress_bar:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs, None)

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                outputs, _ = model(inputs, None)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(loader)
        val_losses.append(val_loss)
        model.train()

        logging.info(f'Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'models/pretrained_clm.pt')
    logging.info("CLM pre-training complete. Model saved to 'models/pretrained_clm.pt'")

    return train_losses, val_losses

def finetune_clm(model, smiles, char_to_idx, device, epochs=10, batch_size=32, lr=0.0001):
    dataset = SMILESDataset(smiles, char_to_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch in progress_bar:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs, None)

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                outputs, _ = model(inputs, None)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(loader)
        val_losses.append(val_loss)
        model.train()

        logging.info(f'Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'models/finetuned_clm.pt')
    logging.info("CLM fine-tuning complete. Model saved to 'models/finetuned_clm.pt'")

    return train_losses, val_losses
