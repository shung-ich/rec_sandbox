import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, num_items, max_len, embed_dim, num_heads, num_layers, dropout):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.embedding = nn.Embedding(num_items, embed_dim)
        self.position_embedding = nn.Embedding(max_len + 100, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=dropout),
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_items)

    def forward(self, input_seq):
        seq_len = input_seq.size(1)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        # (batch_size, max_len, embed_dim)
        item_embed = self.embedding(input_seq)
        pos_embed = self.position_embedding(positions)
        x = self.dropout(item_embed + pos_embed)
        # (batch_size, max_len, embed_dim)
        x = self.transformer(x)
        # (batch_size, max_len, num_items)
        logits = self.fc(x)

        # embedding.weightのshapeは(num_items, embed_dim)
        # (batch_size, max_len, num_items)
        similarity = torch.matmul(x, self.embedding.weight.T)

        
        return similarity

