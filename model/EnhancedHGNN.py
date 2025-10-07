import torch
import torch.nn as nn
from torch import Tensor
from model.EnhancedHGNNConv import EnhancedHGNNConv

class EnhancedHGNN(nn.Module):
    def __init__(
            self,
            node_dim: int,
            emb_dim: int = 128,
            num_layers: int = 2,
            num_heads: int = 4,    # <--- 新增
            attn_dim: int = 128
    ):

        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        self.node_encoder = nn.Linear(node_dim, emb_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            self.layers.append(
                EnhancedHGNNConv(
                    node_in_dim=emb_dim,
                    node_out_dim=emb_dim,
                    hyper_in_dim=emb_dim,
                    hyper_out_dim=emb_dim,
                    drop_rate=0.1,
                    is_last=is_last,
                    num_heads=num_heads,  # <--- 传递参数
                    attn_dim=attn_dim 
                )
            )

    def forward(self, X: Tensor, hyperedge_index: Tensor) -> tuple[Tensor, Tensor]:

        X = self.node_encoder(X)
        Y = self._init_hyper_emb(X, hyperedge_index)
        
        for layer in self.layers:
            X, Y = layer(X, Y, hyperedge_index)

        return X, Y

    def _init_hyper_emb(self, X: Tensor, hyperedge_index: Tensor) -> Tensor:

        node_idx, edge_idx = hyperedge_index
        num_hyperedges = edge_idx.max().item() + 1
        counts = torch.zeros(num_hyperedges, device=X.device)
        counts.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))
        counts = torch.clamp(counts, min=1)
    
        Y = torch.zeros(num_hyperedges, X.size(1), device=X.device)
        Y.scatter_add_(0, edge_idx.unsqueeze(1).expand(-1, X.size(1)), X[node_idx])
        Y = Y / counts.unsqueeze(1)
        
        return Y