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
        x = self.features(x)
        x = self.pool(x)
        features = torch.flatten(x, 1) 

        device = features.device
        num_nodes = features.size(0)
        
        cols = torch.arange(num_nodes, dtype=torch.long, device=device)
        rows = torch.arange(num_nodes, dtype=torch.long, device=device)
        edge_index = torch.stack([rows, cols], dim=0)

        x = self.gat1(features, edge_index)
        x = F.elu(x) 
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Use the new classifier name
        out = self.classifier(x)
        return out