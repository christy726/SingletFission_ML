import torch
import logging
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    train_losses = []
    val_losses = []
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
        
        # Validation
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
    return train_losses, val_losses

def generate_smiles_batch(model, char_to_idx, idx_to_char, start_chars, max_length=100, beam_width=5):
    model.eval()
    device = next(model.parameters()).device
    batch_size = len(start_chars)
    results = [[] for _ in range(batch_size)]
    beams = [[[[c], 0.0, model.init_hidden(1)] for c in start_chars]]  # (sequence, log_prob, hidden)
    
    for _ in range(max_length-1):
        new_beams = []
        for b_idx in range(batch_size):
            current_beams = beams[b_idx]
            if not current_beams:
                new_beams.append([])
                continue
            
            # Prepare inputs
            inputs = torch.tensor([[char_to_idx[b[0][-1]] for b in current_beams]]).T.to(device)
            hidden_states = [b[2] for b in current_beams]
            hidden = (
                torch.cat([h[0] for h in hidden_states], dim=1),
                torch.cat([h[1] for h in hidden_states], dim=1)
            )
            
            # Forward pass
            outputs, new_hidden = model(inputs, hidden)
            probs = torch.softmax(outputs[:, -1], dim=-1).squeeze(1)
            
            # Expand beams
            new_candidates = []
            for beam_idx, (seq, log_p, _) in enumerate(current_beams):
                topk = probs[beam_idx].topk(beam_width)
                for c_idx, prob in zip(topk.indices, topk.values):
                    c = idx_to_char[c_idx.item()]
                    new_log_p = log_p + prob.log()
                    new_seq = seq + [c]
                    new_hidden_part = (
                        new_hidden[0][:, beam_idx:beam_idx+1],
                        new_hidden[1][:, beam_idx:beam_idx+1]
                    )
                    new_candidates.append((new_seq, new_log_p, new_hidden_part))
            
            # Select top K beams
            new_candidates.sort(key=lambda x: -x[1])
            selected = new_candidates[:beam_width]
            new_beams.append(selected)
        beams = new_beams
    
    # Select best beam for each sequence
    final_smiles = []
    for beam in beams:
        if beam:
            best_seq = beam[0][0]
            final_smiles.append(''.join(best_seq))
        else:
            final_smiles.append('')
    return final_smiles

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    logging.info(f'Evaluation Avg Loss: {avg_loss:.4f}')
    return {'avg_loss': avg_loss}