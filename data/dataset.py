import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
import pydicom
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom
from typing import Optional, List, Union, Callable
from PIL import Image
from typing import Optional, Callable, Tuple, Any
from config import MIMIC_CXR_PATH,MIMIC_CXR_JPG_PATH,FILTER_SIZE
from pathlib import Path
import cv2


def simpler_loader(dicom_path):
    # Load DICOM image
    dicom = pydicom.dcmread(dicom_path)
    
    # Get pixel array and convert to float64
    image = dicom.pixel_array.astype(np.float64)

    # image = self.clahe.apply(image)

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image # some images are inverted
    
    # Min-max scale the image to [0, 1] using its own min/max values
    # This preserves the full  dynamic range of the original int16 data.
    image = (image - image.min())/(image.max() - image.min()) 
    return image

class MIMICCXRAdvancedDataset(Dataset):
    def __init__(self, base_dir, x_paths, y_values, classes, transform=None):
        """
        Args:
            base_dir (str): Base directory where DICOM files are stored.
            x_paths (pd.Series): Relative paths to the DICOM files.
            y_values (pd.Series): Class labels.
            classes (list): List of class names.
            transform (callable, optional): Image transformations.
        """
        self.base_dir = base_dir
        self.x_paths = x_paths.reset_index(drop=True)
        self.y_values = y_values.reset_index(drop=True)
        self.classes = classes
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        # Construct full DICOM path
        dicom_path = os.path.join(self.base_dir, self.x_paths.iloc[idx])


        # image, _ = load_and_standardize_cxr_dicom(dicom_path, return_metadata=True)

        # image = improved_cxr_preprocessing(dicom_path, standardization_mode='minmax')

        # image = simpler_loader(dicom_path)

        # Load DICOM image
        dicom = pydicom.dcmread(dicom_path)

        image = dicom.pixel_array.astype(np.float32)

        # Min-max scale the image to [0, 1] using its own min/max values
        # This preserves the full dynamic range of the original int16 data.
        image = (image - image.min()) / (image.max() - image.min())
        # image = image/255.0

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = 1 - image  # some images are inverted




        # Convert to PIL Image in 'F' mode for float32 data.
        # transforms.ToTensor() will correctly handle this 'F' mode PIL image
        # and convert it to a torch.FloatTensor in the [0, 1] range.
        image = Image.fromarray(image, mode='F')

        # Apply transformation (should produce tensor of shape (1, H, W))
        if self.transform:
            # print("Current image shape is ", np.array(image).shape)
            image = self.transform(image)

        # Get label
        label = self.y_values.iloc[idx]
        if isinstance(label, str):
            label = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        elif isinstance(label, (list, np.ndarray)):
            # This branch is for multi-label (e.g., one-hot encoded) situations.
            # For single-label classification with 7 classes, it's usually a single integer.
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label)

        # Ensure label is long for nn.CrossEntropyLoss
        return image, label.long()

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
    


class MultimodalDicomDataset(Dataset):
    """
    PyTorch Dataset for multimodal data combining DICOM images and tabular features.
    
    Args:
        X_data (pd.DataFrame): DataFrame with DICOM file paths in first column and tabular features in remaining columns
        y_data (pd.Series): Target labels (already encoded)
        dicom_col (str): Name of column containing DICOM file paths (default: first column)
        img_transform (callable, optional): Transform to apply to DICOM images
    
    Note: Tabular features are assumed to be already preprocessed/normalized.
    """
    
    def __init__(
        self, 
        X_data: pd.DataFrame,
        y_data: pd.Series,
        dicom_col: Optional[str] = None,
        img_transform: Optional[Callable] = None
    ):
        self.X_data = X_data.reset_index(drop=True)
        self.y_data = y_data.reset_index(drop=True)
        
        # Use first column as DICOM path column if not specified
        self.dicom_col = dicom_col if dicom_col else X_data.columns[0]
        
        # Separate DICOM paths and tabular features
        self.dicom_paths = self.X_data[self.dicom_col].values
        self.tabular_features = self.X_data.drop(columns=[self.dicom_col])
        
        # Store feature names for reference
        self.tabular_feature_names = list(self.tabular_features.columns)
        
        # Convert tabular features to numpy array (already preprocessed)
        self.tabular_data = self.tabular_features.values.astype(np.float32)


        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        
        # Image transforms - updated for float32 'F' mode images
        if img_transform:
            self.img_transform = img_transform
                    
        # Validate data consistency
        assert len(self.X_data) == len(self.y_data), "X_data and y_data must have same length"


        
    def __len__(self) -> int:
        return len(self.X_data)
    
    def load_dicom_image(self, dicom_path: str) -> Image.Image:
        """
        Load and preprocess DICOM file using the same approach as working unimodal dataset.
        
        Args:
            dicom_path (str): Path to DICOM file
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Load DICOM image
            dicom = pydicom.dcmread(dicom_path)
            
            # Get pixel array and convert to float32
            image = dicom.pixel_array.astype(np.float32)

            # Min-max scale the image to [0, 1] using its own min/max values
            # This preserves the full dynamic range of the original int16 data.
            image = (image - image.min()) / (image.max() - image.min())

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                image = 1 - image  # some images are inverted

            # Convert to PIL Image in 'F' mode for float32 data.
            # transforms.ToTensor() will correctly handle this 'F' mode PIL image
            # and convert it to a torch.FloatTensor in the [0, 1] range.
            image = Image.fromarray(image, mode='F')
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"Error loading DICOM file {dicom_path}: {str(e)}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (img_modality_x, img_modality_y, tb_modality_x, tb_modality_y)
                - img_modality_x: Image tensor
                - img_modality_y: Image target (same as tb_modality_y)
                - tb_modality_x: Tabular features tensor
                - tb_modality_y: Target tensor
        """
        # Load and process DICOM image
        dicom_path = Path(MIMIC_CXR_PATH, self.dicom_paths[idx])
        
        # Validate file exists
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        img = self.load_dicom_image(dicom_path)
        img_tensor = self.img_transform(img)
        
        # Get tabular features
        tabular_features = self.tabular_data[idx]
        tb_tensor = torch.tensor(tabular_features, dtype=torch.float32)
        
        # Get target
        target = torch.tensor(self.y_data.iloc[idx], dtype=torch.long)
        
        # Return as (img_modality_x, img_modality_y, tb_modality_x, tb_modality_y)
        return img_tensor, tb_tensor, target
    
    def get_feature_names(self) -> list:
        """Get tabular feature names."""
        return self.tabular_feature_names



class MIMICCXRJPGDataset(Dataset):
    def __init__(self, base_dir, x_paths, y_values, classes, transform=None, apply_clahe=False):
        """
        Args:
            base_dir (str): Base directory where JPG files are stored.
            x_paths (pd.Series): Relative paths to the JPG files.
            y_values (pd.Series): Class labels.
            classes (list): List of class names.
            transform (callable, optional): Image transformations.
            apply_clahe (bool): Whether to apply CLAHE for contrast enhancement.
        """
        self.base_dir = base_dir
        self.x_paths = x_paths.reset_index(drop=True)
        self.y_values = y_values.reset_index(drop=True)
        self.classes = classes
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.apply_clahe = apply_clahe

        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        # Construct full JPG path
        img_path = os.path.join(self.base_dir, self.x_paths.iloc[idx])

        # Load image as grayscale
        image = Image.open(img_path).convert('L')  # 'L' mode is 8-bit grayscale

        if self.apply_clahe:
            # Apply CLAHE using OpenCV
            image_np = np.array(image)
            image_np = self.clahe.apply(image_np)
            image = Image.fromarray(image_np)

        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.y_values.iloc[idx]
        if isinstance(label, str):
            label = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        elif isinstance(label, (list, np.ndarray)):
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label)

        return image, label.long()



class MultimodalDicomJPGDataset(Dataset):
    """
    PyTorch Dataset for multimodal data combining jpg images and tabular features.
    
    Args:
        X_data (pd.DataFrame): DataFrame with jpg file paths in first column and tabular features in remaining columns
        y_data (pd.Series): Target labels (already encoded)
        jpg_col (str): Name of column containing JPG file paths (default: first column)
        img_transform (callable, optional): Transform to apply to DICOM images
    
    Note: Tabular features are assumed to be already preprocessed/normalized.
    """
    
    def __init__(
        self, 
        X_data: pd.DataFrame,
        y_data: pd.Series,
        jpg_col: Optional[str] = None,
        img_transform: Optional[Callable] = None
    ):
        self.X_data = X_data.reset_index(drop=True)
        self.y_data = y_data.reset_index(drop=True)
        
        # Use first column as JPG path column if not specified
        self.jpg_col = jpg_col if jpg_col else X_data.columns[0]
        
        # Separate JPG paths and tabular features
        self.dicom_paths = self.X_data[self.jpg_col].values
        self.tabular_features = self.X_data.drop(columns=[self.jpg_col])
        
        # Store feature names for reference
        self.tabular_feature_names = list(self.tabular_features.columns)
        
        # Convert tabular features to numpy array (already preprocessed)
        self.tabular_data = self.tabular_features.values.astype(np.float32)


        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        
        # Image transforms - updated for float32 'F' mode images
        if img_transform:
            self.img_transform = img_transform
                    
        # Validate data consistency
        assert len(self.X_data) == len(self.y_data), "X_data and y_data must have same length"


        
    def __len__(self) -> int:
        return len(self.X_data)
    
    def load_jpg_image(self, jpg_path: str) -> Image.Image:
        """
        Load and preprocess JPG file.
        
        Args:
            jpg_path (str): Path to JPG file
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Load JPG image
            img = Image.open(jpg_path).convert('L')  # Convert to grayscale
            
            if self.clahe:
                # Apply CLAHE using OpenCV
                img_np = np.array(img)
                img_np = self.clahe.apply(img_np)
                img = Image.fromarray(img_np)
            
            return img
            
        except Exception as e:
            raise RuntimeError(f"Error loading JPG file {jpg_path}: {str(e)}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (img_modality_x, img_modality_y, tb_modality_x, tb_modality_y)
                - img_modality_x: Image tensor
                - img_modality_y: Image target (same as tb_modality_y)
                - tb_modality_x: Tabular features tensor
                - tb_modality_y: Target tensor
        """
        # Load and process JPG image
        jpg_path = Path(MIMIC_CXR_JPG_PATH, self.dicom_paths[idx])
        
        # Validate file exists
        if not os.path.exists(jpg_path):
            raise FileNotFoundError(f"DICOM file not found: {jpg_path}")
        
        img = self.load_jpg_image(jpg_path)
        img_tensor = self.img_transform(img)
        
        # Get tabular features
        tabular_features = self.tabular_data[idx]
        tb_tensor = torch.tensor(tabular_features, dtype=torch.float32)
        
        # Get target
        target = torch.tensor(self.y_data.iloc[idx], dtype=torch.long)
        
        # Return as (img_modality_x, img_modality_y, tb_modality_x, tb_modality_y)
        return img_tensor, tb_tensor, target
    
    def get_feature_names(self) -> list:
        """Get tabular feature names."""
        return self.tabular_feature_names
    
# Example usage:
if __name__ == "__main__":
    # Example of how to use the improved dataset
    pass