import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GATConv

class HybridCNNGAT(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNGAT, self).__init__()
        
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        cnn_out_dim = 1280 

        self.gat1 = GATConv(cnn_out_dim, 256, heads=4, concat=True) 
        self.gat2 = GATConv(1024, 512, heads=1, concat=False) 
        
        # Classifier setup to match state_dict "classifier.1"
        self.classifier = nn.Sequential(
            nn.Identity(),              # index 0
            nn.Linear(512, num_classes) # index 1
        )

    def forward(self, x):
        # 1. Feature Extraction (MobileNet features: [B, 1280, 7, 7])
        x = self.features(x) 
        
        # 2. Reshape for GAT (Treat 7x7 spatial grid as 49 nodes)
        batch_size, channels, h, w = x.size()
        nodes = x.view(batch_size, channels, h * w).permute(0, 2, 1) 
        
        # Flatten nodes for GAT processing
        x_gat = nodes.reshape(-1, channels)

        # 3. Create Graph Edges (Fully connected grid)
        edge_index = self._get_fully_connected_edges(h * w, batch_size, x.device)

        # 4. Graph Attention Layers
        x = self.gat1(x_gat, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # 5. Global Pooling (Average the 49 node features)
        x = x.view(batch_size, h * w, -1).mean(dim=1)

        # 6. Final Classification
        out = self.classifier(x)
        return out

    def _get_fully_connected_edges(self, num_nodes_per_img, batch_size, device):
        all_edges = []
        for i in range(batch_size):
            offset = i * num_nodes_per_img
            base_nodes = torch.arange(num_nodes_per_img, device=device)
            src = base_nodes.repeat_interleave(num_nodes_per_img) + offset
            dst = base_nodes.repeat(num_nodes_per_img) + offset
            all_edges.append(torch.stack([src, dst], dim=0))
        
        return torch.cat(all_edges, dim=1)
