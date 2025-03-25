import torch
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, smiles, char_to_idx, seq_len=100):
        self.smiles = smiles
        self.char_to_idx = char_to_idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        idxs = [self.char_to_idx.get(c, self.char_to_idx['<PAD>']) for c in smi] + [self.char_to_idx['<EOS>']]
        if len(idxs) < self.seq_len:
            idxs += [self.char_to_idx['<PAD>']] * (self.seq_len - len(idxs))
        else:
            idxs = idxs[:self.seq_len]
        return torch.tensor(idxs, dtype=torch.long)

#-----------------------------------------------------------------------------------------------------------------

# import torch
# from torch.utils.data import Dataset

# class SMILESDataset(Dataset):
#     def __init__(self, smiles, char_to_idx, seq_len=100):
#         self.smiles = smiles
#         self.char_to_idx = char_to_idx
#         self.seq_len = seq_len

#     def __len__(self):
#         return len(self.smiles)

#     def __getitem__(self, idx):
#         smi = self.smiles[idx]
#         idxs = [self.char_to_idx.get(c, self.char_to_idx['<PAD>']) for c in smi] + [self.char_to_idx['<EOS>']]
#         if len(idxs) < self.seq_len:
#             idxs += [self.char_to_idx['<PAD>']] * (self.seq_len - len(idxs))
#         else:
#             idxs = idxs[:self.seq_len]
#         return torch.tensor(idxs, dtype=torch.long)