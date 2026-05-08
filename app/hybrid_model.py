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
        
        # CHANGED: Renamed from self.fc to self.classifier to match your .pth file
        # We use nn.Sequential with a placeholder at index 0 to match "classifier.1"
        self.classifier = nn.Sequential(
            nn.Identity(), # index 0
            nn.Linear(512, num_classes) # index 1
        )

   def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        x = self.features(x) 
        # After MobileNet features, x is usually [Batch, 1280, 7, 7]
        
        # 1. Treat each 1x1 spatial location as a node
        # Reshape from [B, C, H, W] -> [B, H*W, C]
        batch_size, channels, h, w = x.size()
        nodes = x.view(batch_size, channels, h * w).permute(0, 2, 1) 
        # Now nodes shape is [Batch, 49 nodes, 1280 features]

        # 2. Flatten for GAT (processing all nodes in the batch)
        # nodes shape becomes [Batch * 49, 1280]
        x_gat = nodes.reshape(-1, channels)

        # 3. Create edges for a 7x7 grid (connecting neighboring parts of the leaf)
        # For simplicity, we can use a "Fully Connected" graph for these 49 nodes
        edge_index = self._get_fully_connected_edges(h * w, batch_size, x.device)

        # 4. Pass through GAT
        x = self.gat1(x_gat, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # 5. Global Pooling: Average the 49 nodes back into 1 vector per image
        # Reshape back to [Batch, 49, 512] then mean over nodes
        x = x.view(batch_size, h * w, -1).mean(dim=1)

        # 6. Final Classification
        out = self.classifier(x)
        return out

    def _get_fully_connected_edges(self, num_nodes_per_img, batch_size, device):
        # This creates edges between all nodes within the SAME image
        # This allows the GAT to see the "context" of the whole leaf
        all_edges = []
        for i in range(batch_size):
            offset = i * num_nodes_per_img
            # Create a fully connected adjacency matrix for one image
            base_nodes = torch.arange(num_nodes_per_img, device=device)
            # Create all possible pairs (source, target)
            src = base_nodes.repeat_interleave(num_nodes_per_img) + offset
            dst = base_nodes.repeat(num_nodes_per_img) + offset
            all_edges.append(torch.stack([src, dst], dim=0))
        
        return torch.cat(all_edges, dim=1)
