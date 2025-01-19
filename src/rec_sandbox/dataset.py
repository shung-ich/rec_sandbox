from torch.utils.data import Dataset, DataLoader

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
        target = seq[-1]
        input_seq = seq[:-1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)