from torch.utils.data import Dataset, DataLoader
import torch

class MovieLensDataset(Dataset):
    def __init__(self, user_seq, max_len):
        self.user_seq = user_seq
        self.max_len = max_len

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        seq = self.user_seq[idx]
        # Padding
        seq = seq[-self.max_len:]
        seq = [0] * (self.max_len - len(seq)) + seq
        original_seq = seq[:-1]
        shift_seq = seq[1:]
        # target = seq[-i]
        # input_seq = seq[:-i]
        return torch.tensor(original_seq, dtype=torch.long), torch.tensor(shift_seq, dtype=torch.long)