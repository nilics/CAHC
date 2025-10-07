import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class HGNNEncoder(nn.Module):
    def __init__(self, node_dim: int, emb_dim: int, num_layers: int):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, emb_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HypergraphConv(emb_dim, emb_dim))
            
    def forward(self, X: torch.Tensor, hyperedge_index: torch.Tensor) -> tuple[torch.Tensor, None]:
        X = self.node_encoder(X)
        for layer in self.layers:
            X = layer(X, hyperedge_index)
            X = F.relu(X)
        return X, None