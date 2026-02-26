import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import math

class BrainRegionGNN(nn.Module):
    """
    Enhanced GNN for predicting NE release from cortical Ca activity.
    Models dynamic connectivity between 12 brain regions using attention mechanisms.
    """
    def __init__(self, 
                 num_regions=12, 
                 gcn_hidden=64,
                 temporal_hidden=128,
                 num_gcn_layers=3,
                 num_gat_heads=8,
                 dropout=0.1,
                 connectivity_type='learned',
                 use_temporal_attention=True):
        super().__init__()
        
        self.num_regions = num_regions
        self.connectivity_type = connectivity_type
        self.use_temporal_attention = use_temporal_attention
        self.gcn_hidden = gcn_hidden
        
        # Input feature transformation  
        self.input_transform = nn.Linear(1, gcn_hidden//2)
        
        # Graph convolution layers with residual connections
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GATConv(gcn_hidden//2, gcn_hidden//num_gat_heads, 
                                     heads=num_gat_heads, dropout=dropout, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(gcn_hidden))
        
        # Subsequent layers
        for _ in range(num_gcn_layers-1):
            self.gcn_layers.append(GATConv(gcn_hidden, gcn_hidden//num_gat_heads, 
                                         heads=num_gat_heads, dropout=dropout, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(gcn_hidden))
        
        # Connectivity learning (for dynamic adjacency)
        if connectivity_type == 'learned':
            # The connectivity decision will be based on the processed features after GCNs
            # After GCN processing, features will be gcn_hidden, so edge features are gcn_hidden * 2
            self.connectivity_mlp = nn.Sequential(
                nn.Linear(gcn_hidden * 2, gcn_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gcn_hidden, 1),
                nn.Sigmoid()
            )
        
        # Regional importance weighting
        self.region_attention = nn.MultiheadAttention(
            embed_dim=gcn_hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal modeling with LSTM + Attention
        self.temporal_lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=temporal_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        if use_temporal_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=temporal_hidden * 2,  # bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Final prediction layers
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(temporal_hidden * 2, temporal_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden, temporal_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden//2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def compute_edge_weights(self, node_features):
        """
        Compute edge weights for learned connectivity
        node_features: [num_nodes, feature_dim]
        """
        num_nodes = node_features.size(0)
        
        # Create all possible edges
        edges = torch.combinations(torch.arange(num_nodes), r=2)
        edge_index = torch.cat([edges, edges.flip(1)], dim=0)  # make undirected
        
        # Compute edge features
        node_i = node_features[edge_index[:, 0]]  # [num_edges, feature_dim]
        node_j = node_features[edge_index[:, 1]]  # [num_edges, feature_dim]
        edge_features = torch.cat([node_i, node_j], dim=1)  # [num_edges, feature_dim*2]
        
        # Compute weights
        weights = self.connectivity_mlp(edge_features).squeeze(-1)  # [num_edges]
        
        return weights, edge_index
    
    def create_correlation_adjacency(self, x, threshold=0.3):
        """
        Create adjacency based on correlation between regions over time
        x: [batch_size, seq_len, num_regions]
        """
        batch_size, seq_len, num_regions = x.shape
        edge_indices = []
        
        for b in range(batch_size):
            # Compute correlation between regions
            data = x[b]  # [seq_len, num_regions]
            corr_matrix = torch.corrcoef(data.T)  # [num_regions, num_regions]
            
            # Apply threshold and create edge list
            adj_matrix = (torch.abs(corr_matrix) > threshold).float()
            adj_matrix.fill_diagonal_(0)  # Remove self-loops
            
            edge_index = adj_matrix.nonzero().T
            edge_indices.append(edge_index)
        
        return edge_indices
    
    def forward(self, x, edge_index=None):
        """
        x: [batch_size, seq_len, num_regions] - Ca activity from 12 brain regions
        Returns: [batch_size, seq_len] - Predicted NE release
        """
        batch_size, seq_len, num_regions = x.shape
        assert num_regions == self.num_regions, f"Expected {self.num_regions} regions, got {num_regions}"
        
        # Transform input features
        x = x.unsqueeze(-1)  # [batch_size, seq_len, num_regions, 1]
        x = x.view(batch_size * seq_len, num_regions, 1)
        x = self.input_transform(x)  # [batch_size * seq_len, num_regions, gcn_hidden//2]
        
        # Create adjacency if not provided
        if edge_index is None:
            if self.connectivity_type == 'correlation':
                input_for_corr = x.view(batch_size, seq_len, num_regions, -1).mean(-1)
                edge_indices = self.create_correlation_adjacency(input_for_corr)
            elif self.connectivity_type == 'learned':
                # For learned connectivity, we'll compute it after initial processing
                edge_indices = None  # Will be computed later
            else:  # fully connected
                edge_indices = [torch.combinations(torch.arange(num_regions), r=2).T for _ in range(batch_size * seq_len)]
        
        # Apply graph convolutions
        gcn_outputs = []
        for b in range(batch_size * seq_len):
            h = x[b]  # [num_regions, gcn_hidden//2]
            
            # Initial GCN layer
            if edge_index is None and self.connectivity_type != 'learned':
                # Select appropriate edge_index for this batch
                if isinstance(edge_indices, list):
                    batch_edge_index = edge_indices[b % len(edge_indices)]
                else:
                    # Create fully connected as fallback
                    batch_edge_index = torch.combinations(torch.arange(num_regions), r=2).T
                    batch_edge_index = torch.cat([batch_edge_index, batch_edge_index.flip(1)], dim=0).T
                batch_edge_index = batch_edge_index.to(h.device)
            elif edge_index is not None:
                batch_edge_index = edge_index
            else:
                # For learned connectivity, start with fully connected
                batch_edge_index = torch.combinations(torch.arange(num_regions), r=2).T
                batch_edge_index = torch.cat([batch_edge_index, batch_edge_index.flip(1)], dim=0).T
                batch_edge_index = batch_edge_index.to(h.device)
            
            # Graph convolutions with residual connections
            for i, (gcn_layer, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
                h_new = gcn_layer(h, batch_edge_index)
                h_new = bn(h_new.view(-1, self.gcn_hidden)).view(-1, self.gcn_hidden)
                h_new = F.relu(h_new)
                
                # Apply learned connectivity after first layer if specified
                if i == 0 and self.connectivity_type == 'learned':
                    # Compute dynamic adjacency based on processed features
                    edge_weights = self.compute_edge_weights(h_new)
                    # For simplicity, we'll keep the current adjacency structure
                    # but this is where you could update batch_edge_index based on weights
                
                # Residual connection (after first layer)
                if i > 0 and h.shape[-1] == h_new.shape[-1]:
                    h = h + h_new
                else:
                    h = h_new
            
            gcn_outputs.append(h)
        
        # Stack and reshape
        gcn_output = torch.stack(gcn_outputs)  # [batch_size * seq_len, num_regions, gcn_hidden]
        gcn_output = gcn_output.view(batch_size, seq_len, num_regions, -1)
        
        # Regional attention - determine which regions are important
        region_features = gcn_output.view(batch_size * seq_len, num_regions, -1)
        attended_features, _ = self.region_attention(
            region_features, region_features, region_features
        )
        
        # Global pooling across regions to get sequence representation
        sequence_features = attended_features.mean(dim=1)  # [batch_size * seq_len, gcn_hidden]
        sequence_features = sequence_features.view(batch_size, seq_len, -1)
        
        # Temporal modeling with LSTM
        temporal_output, _ = self.temporal_lstm(sequence_features)
        temporal_output = self.dropout(temporal_output)
        
        # Temporal attention
        if self.use_temporal_attention:
            attended_temporal, _ = self.temporal_attention(
                temporal_output, temporal_output, temporal_output
            )
            temporal_output = temporal_output + attended_temporal
        
        # Final prediction
        predictions = self.fc_layers(temporal_output)  # [batch_size, seq_len, 1]
        predictions = predictions.squeeze(-1)  # [batch_size, seq_len]
        
        return predictions


class SpatioTemporalGNN(nn.Module):
    """Legacy GNN model for backward compatibility"""
    def __init__(self, num_nodes=12, gcn_hidden=32, rnn_hidden=64, dropout=0.2):
        super().__init__()

        self.num_nodes = num_nodes

        # Graph layers
        self.gcn1 = GCNConv(1, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)

        # Temporal model
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=rnn_hidden,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden, 1)

    def forward(self, x, edge_index=None):
        """
        x: [B, T, N] or [B, T, N, 1]
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # Add feature dimension
        
        B, T, N, F = x.shape
        assert N == self.num_nodes

        # Create simple adjacency if not provided
        if edge_index is None:
            # Fully connected graph
            edge_index = torch.combinations(torch.arange(N), r=2)
            edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).T
            edge_index = edge_index.to(x.device)

        # ---- Apply GCN to all timepoints at once ----
        x = x.reshape(B * T, N, F)  # [B*T, N, 1]

        gcn_out = []
        for batch_idx in range(B * T):
            h = F.relu(self.gcn1(x[batch_idx], edge_index))
            h = F.relu(self.gcn2(h, edge_index))
            h = h.mean(dim=0)  # global mean pool
            gcn_out.append(h)

        gcn_out = torch.stack(gcn_out)
        gcn_out = gcn_out.view(B, T, -1)  # [B, T, gcn_hidden]

        # ---- Temporal modeling ----
        rnn_out, _ = self.gru(gcn_out)
        rnn_out = self.dropout(rnn_out)

        y_pred = self.fc(rnn_out).squeeze(-1)  # [B, T]

        return y_pred


def build_graph_from_corr(ca_data, threshold=0.3):
    """
    Build graph connectivity from correlation matrix with Fisher z-transform
    ca_data: [time, num_regions] or [time, num_regions]
    """
    import numpy as np
    
    # Handle different input shapes
    if ca_data.ndim == 3:  # [batch, time, regions]
        # Use first batch or average across batches
        ca_data = ca_data[0] if ca_data.shape[0] == 1 else ca_data.mean(axis=0)
    
    corr = np.corrcoef(ca_data.T)
    z = fisher_transform(corr)

    # threshold in z space
    adj = (np.abs(z) > threshold).astype(int)
    np.fill_diagonal(adj, 0)

    edge_index = np.array(np.nonzero(adj))
    return torch.tensor(edge_index, dtype=torch.long)


def fisher_transform(r):
    """Fisher z-transformation for correlation values"""
    import numpy as np
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def create_fixed_brain_graph(num_regions=12, connectivity_type='small_world'):
    """
    Create fixed graph connectivity patterns for brain regions
    """
    if connectivity_type == 'small_world':
        # Create a small-world network typical of brain connectivity
        import networkx as nx
        G = nx.watts_strogatz_graph(num_regions, k=4, p=0.3)
        edge_index = torch.tensor(list(G.edges)).T
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
    elif connectivity_type == 'scale_free':
        # Scale-free network
        import networkx as nx
        G = nx.barabasi_albert_graph(num_regions, m=3)
        edge_index = torch.tensor(list(G.edges)).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
    elif connectivity_type == 'ring':
        # Ring topology - each region connected to neighbors
        edges = []
        for i in range(num_regions):
            edges.append([i, (i+1) % num_regions])
            edges.append([i, (i-1) % num_regions])
        edge_index = torch.tensor(edges).T
        
    elif connectivity_type == 'full':
        # Fully connected
        edge_index = torch.combinations(torch.arange(num_regions), r=2)
        edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).T
        
    else:
        raise ValueError(f"Unknown connectivity type: {connectivity_type}")
    
    return edge_index


# Factory function for easy model creation
def create_neuromodulation_gnn(
    model_type='enhanced',
    num_regions=12,
    connectivity_type='learned',
    **kwargs
):
    """
    Factory function to create GNN models for neuromodulation prediction
    
    Args:
        model_type: 'enhanced' or 'legacy'
        num_regions: Number of brain regions (default 12)
        connectivity_type: 'learned', 'correlation', 'small_world', 'scale_free', 'full'
        **kwargs: Additional model parameters
    """
    if model_type == 'enhanced':
        default_params = {
            'gcn_hidden': 64,
            'temporal_hidden': 128,
            'num_gcn_layers': 3,
            'num_gat_heads': 8,
            'dropout': 0.1,
            'use_temporal_attention': True
        }
        default_params.update(kwargs)
        return BrainRegionGNN(
            num_regions=num_regions,
            connectivity_type=connectivity_type,
            **default_params
        )
        
    elif model_type == 'legacy':
        default_params = {
            'gcn_hidden': 32,
            'rnn_hidden': 64,
            'dropout': 0.2
        }
        default_params.update(kwargs)
        return SpatioTemporalGNN(
            num_nodes=num_regions,
            **default_params
        )