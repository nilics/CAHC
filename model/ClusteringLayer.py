import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringPrototypes(nn.Module):
    def __init__(self, n_clusters: int, emb_dim: int, temperature: float = 0.5):
        super().__init__()
        self.prototypes = nn.Parameter(torch.Tensor(n_clusters, emb_dim))
        self.temperature = temperature
        nn.init.xavier_uniform_(self.prototypes.data)

    def forward(self, node_embeds: torch.Tensor) -> torch.Tensor:
        node_embeds_norm = F.normalize(node_embeds, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarity_matrix = torch.matmul(node_embeds_norm, prototypes_norm.T)
        return similarity_matrix / self.temperature