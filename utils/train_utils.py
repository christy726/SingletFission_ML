import torch
import logging
from tqdm import tqdm

def train_model(model, loader, criterion, optimizer, device, epochs=10):
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
        logging.info(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')

def generate_smiles(model, char_to_idx, idx_to_char, start_char='C', max_length=100):
    model.eval()
    device = next(model.parameters()).device  
    with torch.no_grad():
        input_idx = torch.tensor([[char_to_idx[start_char]]], dtype=torch.long).to(device)
        hidden = None
        smiles = start_char
        for _ in range(max_length):
            output, hidden = model(input_idx, hidden)
            probs = torch.softmax(output[0, -1], dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            if next_char == '<PAD>' or next_char == '<EOS>':
                break
            smiles += next_char
            input_idx = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
        return smiles

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            outputs, _ = model(inputs, None)

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    logging.info(f'Evaluation Avg Loss: {avg_loss:.4f}')
    return {'avg_loss': avg_loss}
