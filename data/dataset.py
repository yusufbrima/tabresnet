import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Union, Callable
from typing import Optional, Callable, Tuple, Any
from config import FILTER_SIZE
from pathlib import Path

class TabularDataset(Dataset):
    def __init__(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Example usage:
if __name__ == "__main__":
    # Example of how to use the improved dataset
    pass