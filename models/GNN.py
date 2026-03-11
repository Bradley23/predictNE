import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GNN_NE(nn.Module):
    def __init__(
        self,
        num_nodes=12,
        node_features=1,
        gat_hidden=32,
        gat_heads=4,
        rnn_hidden=64,
        num_gat_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads

        # ----- Spatial GAT layers -----
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GATConv(
                node_features,
                gat_hidden,
                heads=gat_heads,
                concat=True,
                dropout=dropout
            )
        )

        # Additional layers
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(
                GATConv(
                    gat_hidden * gat_heads,
                    gat_hidden,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout
                )
            )

        # Normalize after GAT stack
        self.spatial_norm = nn.LayerNorm(gat_hidden * gat_heads)

        # ----- Temporal model -----
        self.gru = nn.GRU(
            input_size=gat_hidden * gat_heads,
            hidden_size=rnn_hidden,
            batch_first=True
        )

        self.temporal_norm = nn.LayerNorm(rnn_hidden)

        # ----- Output -----
        self.fc_out = nn.Linear(rnn_hidden, 1)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        Vectorized over time.
        x: (B, T, N) - raw Ca per region
        edge_index: (2, E)
        """

        B, T, N = x.shape
        assert N == self.num_nodes

        # ---- Flatten batch x time to process all timepoints at once ----
        # Add feature dim
        xt = x.unsqueeze(-1)                  # (B, T, N, 1)
        xt = xt.reshape(B*T, N, 1)            # (B*T, N, F)

        # Create batch index for pooling
        batch_index = torch.arange(B*T, device=x.device).repeat_interleave(N)

        # ---- Spatial GAT ----
        xt = xt.reshape(B*T*N, 1)             # combine all nodes
        for gat in self.gat_layers:
            xt = gat(xt, edge_index)
            xt = F.elu(xt)
            xt = self.dropout(xt)

        # Graph-level pooling
        pooled = global_mean_pool(xt, batch_index)  # (B*T, D)
        pooled = self.spatial_norm(pooled)

        # ---- Restore batch x time for GRU ----
        gnn_seq = pooled.view(B, T, -1)      # (B, T, D)

        # ---- Temporal GRU ----
        rnn_out, _ = self.gru(gnn_seq)       # (B, T, rnn_hidden)
        rnn_out = self.temporal_norm(rnn_out)

        # ---- NE prediction ----
        ne_pred = self.fc_out(rnn_out)#.squeeze(-1)  # (B, T)

        return ne_pred

def fully_connected_edge_index(num_nodes):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index