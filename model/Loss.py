import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max

class AdvancedScorer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeds: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = edge_idx.max().item() + 1
        mean_pooled = scatter_mean(node_embeds[node_idx], edge_idx, dim=0, dim_size=num_hyperedges)
        max_pooled = scatter_max(node_embeds[node_idx], edge_idx, dim=0, dim_size=num_hyperedges)[0]
        hyperedge_representation = torch.cat([mean_pooled, max_pooled], dim=1)
        scores = self.mlp(hyperedge_representation).squeeze(-1)
        return scores


class Loss(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = AdvancedScorer(embed_dim, hidden_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.clustering_loss_fn = nn.CrossEntropyLoss()

    def compute_infonce_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(sim_matrix.shape[0], device=z1.device)
        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
        return loss

    def forward(self, 
                pos_node_embeds: torch.Tensor, 
                aug_node_embeds: torch.Tensor,
                other_aug_node_embeds: torch.Tensor,
                pos_hyperedge_index: torch.Tensor, 
                neg_hyperedge_index: torch.Tensor,
                clustering_logits: torch.Tensor,
                alpha: float = 1.0, 
                beta: float = 1.0,
                gamma: float = 1.0,
                tau: float = 0.5):

        pos_scores = self.scorer(pos_node_embeds, pos_hyperedge_index)
        neg_scores = self.scorer(pos_node_embeds, neg_hyperedge_index)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        pos_loss = self.bce_loss(pos_scores, pos_labels)
        neg_loss = self.bce_loss(neg_scores, neg_labels)

        structure_loss = pos_loss + neg_loss
        infonce_loss = self.compute_infonce_loss(aug_node_embeds, other_aug_node_embeds, tau)

        with torch.no_grad():
            pseudo_labels = torch.argmax(clustering_logits, dim=1)
        clustering_loss = self.clustering_loss_fn(clustering_logits, pseudo_labels)
        total_loss = alpha * structure_loss + beta * infonce_loss + gamma * clustering_loss

        metrics = {
            'structure_loss': structure_loss.item(),
            'infonce_loss': infonce_loss.item(),
            'clustering_loss': clustering_loss.item(),
            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item(),
            'avg_pos_score': pos_scores.mean().item() if pos_scores.numel() > 0 else 0,
            'avg_neg_score': neg_scores.mean().item() if neg_scores.numel() > 0 else 0,
        }
        
        return total_loss, metrics