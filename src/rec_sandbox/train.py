import torch
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input_seq, target in tqdm(dataloader, desc="Training"):
        input_seq, target = input_seq.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(input_seq)
        logits = logits[:, -1, :]  # 最後の出力のみ
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_seq, target in tqdm(dataloader, desc="Evaluating"):
            input_seq, target = input_seq.to(device), target.to(device)
            logits = model(input_seq)
            logits = logits[:, -1, :]  # 最後の出力のみ
            loss = criterion(logits, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

