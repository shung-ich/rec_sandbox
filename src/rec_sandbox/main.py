import os
from pathlib import Path 

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

from preprocess import preprocess_movielens
from dataset import MovieLensDataset
from sasrec import SASRec
from train import train, evaluate

def main():
    max_len = 50
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.2
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_seq = preprocess_movielens()
    train_seq, test_seq = train_test_split(user_seq, test_size=0.2, random_state=42)

    train_dataset = MovieLensDataset(train_seq, max_len=max_len)
    test_dataset = MovieLensDataset(test_seq, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_items = max(map(max, user_seq)) + 1

    # モデルもdeviceに送る必要がある
    model = SASRec(num_items, max_len=max_len, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()