import torch
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input_seq, target in tqdm(dataloader, desc="Training"):
        input_seq, target = input_seq.to(device), target.to(device)
        optimizer.zero_grad()
        similarity = model(input_seq)
        top_indices = torch.argmax(similarity, dim=-1)
        # print(top_indices.shape)
        # print(top_indices[0])
        # print(target.shape)
        # print(target[0])

        print(similarity.shape, target.shape)
        target_one_hot = torch.zeros(target.size(0), target.size(1), similarity.size(-1)).to(device)
        target_one_hot.scatter_(2, target.unsqueeze(-1), 1)
        print(similarity.shape, target_one_hot.shape)
        loss = criterion(similarity, target_one_hot)
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
            similarity = model(input_seq)
            target_one_hot = torch.zeros(target.size(0), target.size(1), similarity.size(-1)).to(device)
            target_one_hot.scatter_(2, target.unsqueeze(-1), 1)
            loss = criterion(similarity, target_one_hot)
            
            # logits = model(input_seq)
            # logits = logits[:, -1, :]  # 最後の出力のみ
            # loss = criterion(logits, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

