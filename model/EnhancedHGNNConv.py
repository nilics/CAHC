import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import softmax

class EnhancedHGNNConv(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int,
        hyper_in_dim: int,
        hyper_out_dim: int,
        bias: bool = True,
        drop_rate: float = 0.4,
        is_last: bool = False,
        num_heads: int = 4,
        attn_dim: int = 128 
    ):
        super().__init__()
        self.is_last = is_last
        self.num_heads = num_heads


        self.node_v_lin = nn.Linear(node_in_dim, node_out_dim, bias=bias)
        self.hyper_v_lin = nn.Linear(hyper_in_dim, hyper_out_dim, bias=bias)


        self.attn_dim = attn_dim
        self.head_dim = self.attn_dim // num_heads

        self.node_attn_proj = nn.Linear(node_out_dim, self.attn_dim, bias=False)
        self.hyper_attn_proj = nn.Linear(hyper_out_dim, self.attn_dim, bias=False)


        self.attn_vec_e2n = Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        self.attn_vec_n2e = Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.node_norm = nn.LayerNorm(node_out_dim)
        self.hyper_norm = nn.LayerNorm(hyper_out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_v_lin.weight)
        nn.init.xavier_uniform_(self.hyper_v_lin.weight)
        nn.init.xavier_uniform_(self.node_attn_proj.weight)
        nn.init.xavier_uniform_(self.hyper_attn_proj.weight)
        nn.init.xavier_uniform_(self.attn_vec_e2n)
        nn.init.xavier_uniform_(self.attn_vec_n2e)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, hyperedge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        num_nodes, num_edges = X.size(0), Y.size(0)

        X_trans = self.node_v_lin(X)
        Y_trans = self.hyper_v_lin(Y)
        X_proj = self.node_attn_proj(X_trans)
        Y_proj = self.hyper_attn_proj(Y_trans)
        X_proj_mhead = X_proj.view(num_nodes, self.num_heads, self.head_dim)
        Y_proj_mhead = Y_proj.view(num_edges, self.num_heads, self.head_dim)


        inc_t = SparseTensor(row=hyperedge_index[1], col=hyperedge_index[0],sparse_sizes=(num_edges, num_nodes))
        adj_t = inc_t.t()

        node_queries = X_proj_mhead[adj_t.storage.row()]
        hyper_keys = Y_proj_mhead[adj_t.storage.col()]

        attn_input_e2n = torch.cat([node_queries, hyper_keys], dim=-1)
        scores_e2n = (attn_input_e2n * self.attn_vec_e2n).sum(dim=-1)
        scores_e2n = F.leaky_relu(scores_e2n, 0.2)

        X_agg_list = []
        for head in range(self.num_heads):
            attn_weights = softmax(scores_e2n[:, head] / self.temperature, adj_t.storage.row(), num_nodes=num_nodes)
            attn_adj_t_head = adj_t.set_value(attn_weights, layout='coo')
            X_agg_head = matmul(attn_adj_t_head, Y_trans, reduce='sum')
            X_agg_list.append(X_agg_head)
        X_agg = torch.mean(torch.stack(X_agg_list, dim=0), dim=0)


        hyper_queries = Y_proj_mhead[inc_t.storage.row()]
        node_keys = X_proj_mhead[inc_t.storage.col()]
        attn_input_n2e = torch.cat([hyper_queries, node_keys], dim=-1)
        scores_n2e = (attn_input_n2e * self.attn_vec_n2e).sum(dim=-1)
        scores_n2e = F.leaky_relu(scores_n2e, 0.2)
        
        Y_agg_list = []
        for head in range(self.num_heads):
            attn_weights = softmax(scores_n2e[:, head] / self.temperature, inc_t.storage.row(), num_nodes=num_edges)
            attn_inc_t_head = inc_t.set_value(attn_weights, layout='coo')
            Y_agg_head = matmul(attn_inc_t_head, X_trans, reduce='sum')
            Y_agg_list.append(Y_agg_head)
        Y_agg = torch.mean(torch.stack(Y_agg_list, dim=0), dim=0)


        X_final = self.node_norm(X_trans + X_agg)
        Y_final = self.hyper_norm(Y_trans + Y_agg)

        if not self.is_last:
            X_final = self.act(X_final)
            Y_final = self.act(Y_final)
            X_final = self.drop(X_final)
            Y_final = self.drop(Y_final)
        
        return X_final, Y_final