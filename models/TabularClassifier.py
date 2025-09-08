import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactResidualBlock(nn.Module):
    """Lightweight residual block for tabular data"""
    def __init__(self, dim, dropout=0.3):
        super(CompactResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out = out + residual  # Skip connection
        return F.relu(out)


class AdvancedTabularClassifier(nn.Module):
    """Improved version with flexible architecture but kept compact"""
    def __init__(self, input_dim, output_dim, hidden_dim=None, 
                 num_blocks=2, dropout=0.3, use_reduction=False):
        super(AdvancedTabularClassifier, self).__init__()
        
        # Default to input_dim for hidden dimension
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks (1-4 blocks max to keep it compact)
        num_blocks = min(num_blocks, 3)  # Limit depth
        self.residual_blocks = nn.ModuleList([
            CompactResidualBlock(hidden_dim, dropout) 
            for _ in range(num_blocks)
        ])
        
        # Optional intermediate reduction layer
        self.use_reduction = use_reduction
        if use_reduction:
            reduction_dim = hidden_dim // 2
            self.reduction_layer = nn.Sequential(
                nn.Linear(hidden_dim, reduction_dim),
                nn.BatchNorm1d(reduction_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            final_dim = reduction_dim
        else:
            final_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(final_dim, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input projection
        out = self.input_layer(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Optional reduction
        if self.use_reduction:
            out = self.reduction_layer(out)
        
        # Output
        return self.output_layer(out)

if __name__ == "__main__":
    pass