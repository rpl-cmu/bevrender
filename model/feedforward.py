import torch.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, dropout=0.0):
        super().__init__()

        self.dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.ffn(x)
