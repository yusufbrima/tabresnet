import os 
from pathlib import Path 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIG_PATH
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import gzip
import shutil
from pathlib import Path
import pandas as pd
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyhealth.medcode import InnerMap
from config import MIMIC_CXR_PATH, MIMIC_IV_ED_PATH, OUTPUT_PATH
from pyhealth.medcode import InnerMap
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import textwrap
import datetime


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj


def get_class_weights(y, strategy="inverse", beta=0.9999):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y (array-like): training labels
        strategy (str): "inverse" | "effective" | "median" | "none"
        beta (float): smoothing factor for 'effective' strategy
    
    Returns:
        np.array: weights aligned with sorted unique classes
        np.array: unique classes
    """
    unique_classes, counts = np.unique(y, return_counts=True)

    if strategy == "inverse":
        weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)

    elif strategy == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(unique_classes)  # normalize like sklearn

    elif strategy == "median":
        freq = counts / np.sum(counts)
        median_freq = np.median(freq)
        weights = median_freq / freq

    elif strategy in ["none", "noweighting"]:
        weights = np.ones_like(unique_classes, dtype=np.float32)

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'inverse', 'effective', 'median', or 'none'.")

    return np.array(weights, dtype=np.float32), unique_classes


# def get_class_weights(y, strategy="inverse", beta=0.9999):
#     """
#     Compute class weights for imbalanced datasets.
    
#     Args:
#         y (array-like): training labels
#         strategy (str): "inverse" | "effective" | "median"
#         beta (float): smoothing factor for 'effective' strategy
    
#     Returns:
#         np.array: weights aligned with sorted unique classes
#         np.array: unique classes
#     """
#     unique_classes, counts = np.unique(y, return_counts=True)

#     if strategy == "inverse":
#         weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)

#     elif strategy == "effective":
#         effective_num = 1.0 - np.power(beta, counts)
#         weights = (1.0 - beta) / effective_num
#         weights = weights / np.sum(weights) * len(unique_classes)  # normalize like sklearn

#     elif strategy == "median":
#         freq = counts / np.sum(counts)
#         median_freq = np.median(freq)
#         weights = median_freq / freq

#     else:
#         raise ValueError(f"Unknown strategy: {strategy}. Use 'inverse', 'effective', or 'median'.")

#     return np.array(weights, dtype=np.float32), unique_classes

def compute_class_weights_effective_num(y, beta=0.9999):
    """
    Class-Balanced Loss Based on Effective Number of Samples (Cui et al. 2019)
    https://arxiv.org/abs/1901.05555

    Args:
        y (array-like): training labels
        beta (float): hyperparameter close to 1.0 (e.g., 0.9–0.9999).
                      Larger beta → smoother weights.

    Returns:
        dict {class_label: weight}
    """
    classes, counts = np.unique(y, return_counts=True)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(classes)  # normalize like sklearn
    return dict(zip(classes, weights))


def compute_class_weights_median_freq(y):
    """
    Median Frequency Balancing (commonly used in semantic segmentation).
    w_c = median_freq / freq_c

    Args:
        y (array-like): training labels

    Returns:
        dict {class_label: weight}
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / np.sum(counts)
    median_freq = np.median(freq)
    weights = median_freq / freq
    return dict(zip(classes, weights))


def quantify_dataset_imbalance(class_counts=None, class_weights=None):
    """
    Quantify dataset imbalance using class counts or class weights.

    Args:
        class_counts (array-like, optional): Number of samples per class.
        class_weights (array-like, optional): Class weights, typically inverse to class frequency.

    Returns:
        dict: Dictionary containing:
            - 'cv_class_weights': Coefficient of variation of class weights (None if class_weights not provided)
            - 'imbalance_ratio': Ratio of max to min class counts (None if class_counts not provided)
            - 'entropy': Entropy of class distribution (None if class_counts not provided)
    """
    results = {
        'cv_class_weights': None,
        'imbalance_ratio': None,
        'entropy': None
    }

    if class_weights is not None:
        class_weights = np.array(class_weights)
        mean_weight = np.mean(class_weights)
        std_weight = np.std(class_weights)
        results['cv_class_weights'] = std_weight / mean_weight if mean_weight != 0 else None

    if class_counts is not None:
        class_counts = np.array(class_counts)
        # Avoid division by zero in imbalance ratio
        if np.min(class_counts) > 0:
            results['imbalance_ratio'] = np.max(class_counts) / np.min(class_counts)
        else:
            results['imbalance_ratio'] = None

        proportions = class_counts / class_counts.sum()
        # Calculate entropy, avoid log(0) by masking zero proportions
        entropy = -np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0]))
        results['entropy'] = entropy

    return results



def subsample_df(df, target_col, n_per_class, random_state=42):
    def sampler(x):
        print(f"Sampling {min(n_per_class, len(x))} from class {x.name} of size {len(x)}")
        return x.sample(n=min(n_per_class, len(x)), random_state=random_state)
    sampled_df = df.groupby(target_col, group_keys=False).apply(sampler).reset_index(drop=True)
    print("Sampled df shape:", sampled_df.shape)
    return sampled_df

def file_name_timestamp_generator():
    """Generates a unique timestamp for filenames"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    postfix_str = f"{timestamp}"
    return postfix_str
    

def plot_one_sample_per_class(dataset, classes, lookup, save=True, show=False):
    found = {}
    max_classes = len(classes)

    # Collect one sample per class
    for img, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label not in found:
            found[label] = img
        if len(found) == max_classes:
            break

    # Create subplots
    fig, axes = plt.subplots(1, max_classes, figsize=(3 * max_classes, 4))

    if max_classes == 1:
        axes = [axes]

    for i, (label, img_tensor) in enumerate(sorted(found.items())):
        img = img_tensor.squeeze(0).numpy()  # (1, H, W) → (H, W)
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        class_title = lookup_icd_code(icd_code=lookup[classes[label]])[0]

        # print("Current title: ", class_title)
        
        # Wrap long titles
        wrapped_title = "\n".join(textwrap.wrap(class_title, width=20))
        ax.set_title(wrapped_title, fontsize=16)
        ax.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(f"{FIG_PATH}/random_samples.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.show()


def plot_batch(images, labels, classes, n=6, save=False, show=True):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        img = images[i].squeeze(0).numpy()  # (1, H, W) -> (H, W)
        label = labels[i]
        if hasattr(label, 'item'):
            label_text = classes[label.item()] if isinstance(label.item(), int) else str(label)
        else:
            label_text = str(label)

        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label_text}")
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(f"{FIG_PATH}/samples.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def preprocess_mimic_data_advanced(
    output_path, 
    filename='mimic_multimodal_image_centric_advanced_streamlined.csv',
    filter_size=6000,
    target_col='icd_code_broad',
    impute_missing=False,
    staging=True,
    sub_sample = None,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    verbose = False
):
    """
    Complete preprocessing pipeline for MIMIC multimodal data.
    
    Parameters:
    -----------
    output_path : str
        Path to the directory containing the data file
    filename : str
        Name of the CSV file to load
    filter_size : int
        Number of rows to filter (passed to load_data)
    target_col : str
        Name of the target column
    impute_missing : bool
        Whether to impute missing values (True) or drop rows (False)
    sub_sample : float
        Whether to create a subset of the dataset for faster experimentation [0,1]
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of training data for validation set
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Contains all processed data splits and metadata
        - 'X_train': Training features (including path column)
        - 'X_val': Validation features (including path column)
        - 'X_test': Test features (including path column)
        - 'y_train': Training targets
        - 'y_val': Validation targets
        - 'y_test': Test targets
        - 'scaler': Fitted StandardScaler object
        - 'class_distribution': Target class distribution
        - 'feature_columns': List of feature column names
        - 'dropped_columns': List of columns that were dropped
        - 'encoded_columns': List of columns that were one-hot encoded
    """
    
    print("="*60)
    print("STARTING MIMIC DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    print(f"Step 1: Loading data...")
    df_path = os.path.join(output_path, filename)
    df = load_data(df_path, filter=filter_size, target_col=target_col)
    print(f"Loaded data shape: {df.shape}")
    
    # Step 2: Drop unnecessary columns (excluding 'path' column)
    print(f"\nStep 2: Dropping unnecessary columns...")
    columns_to_discard = [
        'dicom_id', 'subject_id', 'study_id', 'stay_id', 'hadm_id',
        'icd_code', 'icd_title', 'icd_version','disposition'
    ]  # Removed 'path' from this list
    
    # Only drop columns that actually exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_discard if col in df.columns]
    missing_columns = [col for col in columns_to_discard if col not in df.columns]
    
    if existing_columns_to_drop:
        df = drop_columns(df, existing_columns_to_drop)
        print(f"Dropped columns: {existing_columns_to_drop}")
    
    if missing_columns:
        print(f"Columns not found (skipped): {missing_columns}")
    
    print(f"Data shape after dropping columns: {df.shape}")
    print(f"Note: 'path' column retained for output data frames")
    
    # Step 3: Handle missing values
    print(f"\nStep 3: Handling missing values...")
    print(f"Imputation strategy: {'Impute' if impute_missing else 'Drop rows'}")
    
    initial_rows = len(df)
    df_original = df[target_col].copy()
    df = handle_missing_values(df, impute=impute_missing, target_col=target_col)
    final_rows = len(df)
    
    if not impute_missing:
        print(f"Rows before/after handling missing values: {initial_rows} -> {final_rows}")
    
    # Step 4: Check class distribution
    print(f"\nStep 4: Analyzing target variable distribution...")
    class_distribution = df[target_col].value_counts()
    print(f"Class distribution for '{target_col}':")
    for class_name, count in class_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Step 5: Feature normalization and encoding
    print(f"\nStep 5: Feature preprocessing...")
    
    # Define columns to encode (categorical) - path is excluded from encoding
   
    columns_to_encode = ['gender', target_col, 'race', 'arrival_transport', 'pain']

    # Only include columns that exist in the dataframe
    existing_encode_cols = [col for col in columns_to_encode if col in df.columns]
    missing_encode_cols = [col for col in columns_to_encode if col not in df.columns]
    
    if missing_encode_cols:
        print(f"Encoding columns not found (skipped): {missing_encode_cols}")
    
    # Normalize numerical columns (except target, categorical columns, and path)
    print("Class count before subsampling\n", df[target_col].value_counts())
    if sub_sample is not None:
        # df = subsample_df(df, target_col=target_col, n_per_class=sub_sample)
        df = stratified_subsample(df, label_col=target_col, frac=sub_sample)
    print("Normalizing numerical features...")
    df_normalized = df.copy()

    # Exclude 'path' from normalization
    exclude_from_normalization = existing_encode_cols + ['path']
    
    for column in df.columns:
        if (column not in exclude_from_normalization and 
            df[column].dtype in ['int64', 'float64'] and 
            df[column].abs().max() != 0):
            
            df_normalized[column] = df[column] / df[column].abs().max()
    
    numerical_cols_normalized = [col for col in df.columns 
                                if col not in exclude_from_normalization 
                                and df[col].dtype in ['int64', 'float64']]
    print(f"Normalized {len(numerical_cols_normalized)} numerical columns")
    
    # One-hot encode categorical columns (path is preserved as-is)
    print("Applying one-hot encoding...")
    df_encoded = one_hot_encode(df_normalized, existing_encode_cols, target_col=target_col, staging=staging)
    print(f"Data shape after encoding: {df_encoded.shape}")
    
    # Step 6: Create train/test split
    print(f"\nStep 6: Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_encoded, target_col, test_size=test_size, random_state=random_state
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 7: Scale numerical features (excluding path column)
    print(f"\nStep 7: Scaling numerical features...")
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 'path' from numerical columns if it somehow got included
    if 'path' in numerical_columns:
        numerical_columns.remove('path')
    
    print(f"Found {len(numerical_columns)} numerical columns to scale")
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numerical_columns:
        X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
        print("Numerical features scaled using StandardScaler")
    
    # Step 8: Create validation split
    print(f"\nStep 8: Creating validation split...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train
    )
    
    print(f"Final splits:")
    print(f"  Training: {X_train_final.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test_scaled.shape}")
    
    # Verify path column is present in all splits
    path_in_train = 'path' in X_train_final.columns
    path_in_val = 'path' in X_val.columns
    path_in_test = 'path' in X_test_scaled.columns
    
    print(f"\nPath column verification:")
    print(f"  In training set: {path_in_train}")
    print(f"  In validation set: {path_in_val}")
    print(f"  In test set: {path_in_test}")
    
    # Summary
    print(f"\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED")
    print("="*60)
    print(f"Original data: {initial_rows} rows")
    print(f"Final data: {len(df_encoded)} rows")
    print(f"Features: {X_train_final.shape[1]} (including path column)")
    print(f"Target classes: {len(class_distribution)}")
    
    # Return comprehensive results
    return {
        'X_train': X_train_final,
        'X_val': X_val,
        'X_test': X_test_scaled,
        'y_train': y_train_final,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'class_distribution': class_distribution,
        'feature_columns': X_train_final.columns.tolist(),
        'dropped_columns': existing_columns_to_drop,
        'encoded_columns': existing_encode_cols,
        'numerical_columns': numerical_columns,
        'original_shape': (initial_rows, df.shape[1] + len(existing_columns_to_drop)),
        'final_shape': df_encoded.shape,
        'final_df': df_encoded,
        'df_original': df_original
    }

def preprocess_mimic_data(
    output_path, 
    filename='mimic_multimodal_image_centric_advanced_streamlined.csv',
    filter_size=6000,
    target_col='icd_code_broad',
    impute_missing=False,
    test_size=0.2,
    val_size=0.2,
    random_state=42
):
    """
    Complete preprocessing pipeline for MIMIC multimodal data.
    
    Parameters:
    -----------
    output_path : str
        Path to the directory containing the data file
    filename : str
        Name of the CSV file to load
    filter_size : int
        Number of rows to filter (passed to load_data)
    target_col : str
        Name of the target column
    impute_missing : bool
        Whether to impute missing values (True) or drop rows (False)
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of training data for validation set
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Contains all processed data splits and metadata
        - 'X_train': Training features
        - 'X_val': Validation features  
        - 'X_test': Test features
        - 'y_train': Training targets
        - 'y_val': Validation targets
        - 'y_test': Test targets
        - 'scaler': Fitted StandardScaler object
        - 'class_distribution': Target class distribution
        - 'feature_columns': List of feature column names
        - 'dropped_columns': List of columns that were dropped
        - 'encoded_columns': List of columns that were one-hot encoded
    """
    
    print("="*60)
    print("STARTING MIMIC DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    print(f"Step 1: Loading data...")
    df_path = os.path.join(output_path, filename)
    df = load_data(df_path, filter=filter_size)
    print(f"Loaded data shape: {df.shape}")
    
    # Step 2: Drop unnecessary columns
    print(f"\nStep 2: Dropping unnecessary columns...")
    columns_to_discard = [
        'dicom_id', 'subject_id', 'study_id', 'path', 'stay_id', 'hadm_id',
        'icd_code', 'icd_title', 'icd_version'
    ]
    
    # Only drop columns that actually exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_discard if col in df.columns]
    missing_columns = [col for col in columns_to_discard if col not in df.columns]
    
    if existing_columns_to_drop:
        df = drop_columns(df, existing_columns_to_drop)
        print(f"Dropped columns: {existing_columns_to_drop}")
    
    if missing_columns:
        print(f"Columns not found (skipped): {missing_columns}")
    
    print(f"Data shape after dropping columns: {df.shape}")
    
    # Step 3: Handle missing values
    print(f"\nStep 3: Handling missing values...")
    print(f"Imputation strategy: {'Impute' if impute_missing else 'Drop rows'}")
    
    initial_rows = len(df)
    df = handle_missing_values(df, impute=impute_missing, target_col=target_col)
    final_rows = len(df)
    
    if not impute_missing:
        print(f"Rows before/after handling missing values: {initial_rows} -> {final_rows}")
    
    # Step 4: Check class distribution
    print(f"\nStep 4: Analyzing target variable distribution...")
    class_distribution = df[target_col].value_counts()
    print(f"Class distribution for '{target_col}':")
    for class_name, count in class_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Step 5: Feature normalization and encoding
    print(f"\nStep 5: Feature preprocessing...")
    
    # Define columns to encode (categorical)
    columns_to_encode = ['gender', target_col, 'race', 'arrival_transport', 'pain']
    # Only include columns that exist in the dataframe
    existing_encode_cols = [col for col in columns_to_encode if col in df.columns]
    missing_encode_cols = [col for col in columns_to_encode if col not in df.columns]
    
    if missing_encode_cols:
        print(f"Encoding columns not found (skipped): {missing_encode_cols}")
    
    # Normalize numerical columns (except target and categorical columns)
    print("Normalizing numerical features...")
    df_normalized = df.copy()
    
    for column in df.columns:
        if (column not in existing_encode_cols and 
            df[column].dtype in ['int64', 'float64'] and 
            df[column].abs().max() != 0):
            
            df_normalized[column] = df[column] / df[column].abs().max()
            
    print(f"Normalized {len([col for col in df.columns if col not in existing_encode_cols and df[col].dtype in ['int64', 'float64']])} numerical columns")
    
    # One-hot encode categorical columns
    print("Applying one-hot encoding...")
    df_encoded = one_hot_encode(df_normalized, existing_encode_cols, target_col=target_col)
    print(f"Data shape after encoding: {df_encoded.shape}")
    
    # Step 6: Create train/test split
    print(f"\nStep 6: Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_encoded, target_col, test_size=test_size, random_state=random_state
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 7: Scale numerical features
    print(f"\nStep 7: Scaling numerical features...")
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numerical_columns)} numerical columns to scale")
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numerical_columns:
        X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
        print("Numerical features scaled using StandardScaler")
    
    # Step 8: Create validation split
    print(f"\nStep 8: Creating validation split...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train
    )
    
    print(f"Final splits:")
    print(f"  Training: {X_train_final.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test_scaled.shape}")
    
    # Summary
    print(f"\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED")
    print("="*60)
    print(f"Original data: {initial_rows} rows")
    print(f"Final data: {len(df_encoded)} rows")
    print(f"Features: {X_train_final.shape[1]}")
    print(f"Target classes: {len(class_distribution)}")
    
    # Return comprehensive results
    return {
        'X_train': X_train_final,
        'X_val': X_val,
        'X_test': X_test_scaled,
        'y_train': y_train_final,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'class_distribution': class_distribution,
        'feature_columns': X_train_final.columns.tolist(),
        'dropped_columns': existing_columns_to_drop,
        'encoded_columns': existing_encode_cols,
        'numerical_columns': numerical_columns,
        'original_shape': (initial_rows, df.shape[1] + len(existing_columns_to_drop)),
        'final_shape': df_encoded.shape
    }

# Convenience function for common usage
def quick_preprocess_mimic(output_path, impute=False, filter_size=6000):
    """
    Quick preprocessing with common defaults
    """
    return preprocess_mimic_data(
        output_path=output_path,
        impute_missing=impute,
        filter_size=filter_size
    )





def handle_missing_values(df, impute=True, target_col='icd_code_broad', verbose=False):
    """
    Handle missing values in a DataFrame by either imputing or dropping rows.
    Always drops rows where target variable is missing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to process
    impute : bool, default=True
        If True, impute missing values using SimpleImputer for features
        If False, drop rows with missing values
    target_col : str, default='icd_code_broad'
        The target column name - rows with missing target will always be dropped
    
    Returns:
    --------
    pandas.DataFrame
        The processed DataFrame
    """
    
    # First, always drop rows where target variable is missing
    if target_col in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=[target_col])
        target_dropped = initial_rows - len(df)
        if target_dropped > 0:
            print(f"Dropped {target_dropped} rows with missing target variable '{target_col}'")
    
    # Check for missing values in remaining data
    missing_values = check_missing_values(df)
    
    if not missing_values.empty:
        if verbose:
            print("Missing values found in the following columns:")
            print(missing_values)
        
        if impute:
            if verbose:
                print("\nApplying imputation...")
            
            # Create a copy to avoid modifying the original DataFrame
            df_processed = df.copy()
            
            # Separate numerical and categorical columns (excluding target)
            feature_cols = [col for col in df_processed.columns if col != target_col]
            numerical_cols = df_processed[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_processed[feature_cols].select_dtypes(include=['object']).columns.tolist()
            
            # Impute numerical columns with median
            if numerical_cols:
                num_imputer = SimpleImputer(strategy='median')
                df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
                if verbose:
                    print(f"Imputed numerical columns {numerical_cols} with median values")
            
            # Impute categorical columns with most frequent value
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
                if verbose:
                    print(f"Imputed categorical columns {categorical_cols} with most frequent values")
            if verbose:
                print("Imputation completed successfully!")
            return df_processed
            
        else:
            if verbose:
                print("\nDropping rows with missing values...")
            df_processed = df.dropna()
            rows_dropped = len(df) - len(df_processed)
            if verbose:
                print(f"Dropped {rows_dropped} rows with missing values")
                print(f"Remaining rows: {len(df_processed)}")
            return df_processed
    
    else:
        if verbose:
            print("No missing values found.")
        return df.copy()



def get_icd_code(icd_lookup, index_key):
    return icd_lookup.get(index_key, None)  # returns None if key not found


def lookup_icd_code(icd_code, icd_version=None):
    """
    Lookup ICD code using InnerMap. If version is not specified or lookup fails,
    tries ICD9 first, then ICD10.
    
    Args:
        icd_code: The ICD code to lookup
        icd_version: Optional version (9 or 10). If None, tries both versions.
    
    Returns:
        Tuple of (result, version_used) or (None, None) if not found
    """
    icd_code = str(icd_code).strip()  # Ensure icd_code is a string and stripped of whitespace
    
    def try_icd9():
        try:
            icd9cm = InnerMap.load("ICD9CM")
            result = icd9cm.lookup(icd_code)
            return result, '9'
        except (KeyError, Exception):
            return None, None
    
    def try_icd10():
        try:
            icd10cm = InnerMap.load("ICD10CM")
            result = icd10cm.lookup(icd_code)
            return result, '10'
        except (KeyError, Exception):
            return None, None
    
    # If version is specified, try that version first
    if icd_version is not None:
        try:
            version_str = str(int(icd_version))
            if version_str == '9':
                result, used_version = try_icd9()
                if result is not None:
                    return result, used_version
                # If specified version fails, try the other version
                return try_icd10()
            elif version_str == '10':
                result, used_version = try_icd10()
                if result is not None:
                    return result, used_version
                # If specified version fails, try the other version
                return try_icd9()
            else:
                print(f"Unsupported ICD version: {icd_version}, trying both versions")
        except ValueError:
            print(f"Invalid version format: {icd_version}, trying both versions")
    
    # Try ICD9 first, then ICD10
    result, used_version = try_icd9()
    if result is not None:
        return result, used_version
    
    result, used_version = try_icd10()
    if result is not None:
        return result, used_version
    
    # Code not found in either version
    return None, None




# Create a function to load images for training
def load_image_for_training(row, base_path=MIMIC_CXR_PATH):
    """
    Function to load DICOM image for a given row
    Usage: image = load_image_for_training(training_df.iloc[0])
    """
    full_path = os.path.join(base_path, 'files', row['path'])
    # Add your DICOM loading logic here
    # e.g., using pydicom: dcm = pydicom.dcmread(full_path)
    return full_path


def lookup_icd_code_with_ancestor(icd_version, icd_code):
    """
    Lookup ICD code using InnerMap based on version with error handling.
    """
    icd_code = str(icd_code).strip()  # Ensure icd_code is a string and stripped of whitespace
    try:
        if str(int(icd_version)) == '9':
            icd9cm = InnerMap.load("ICD9CM")
            ancestor = icd9cm.get_ancestors(icd_code)[1] # Get the  closest ancestor
            ancestor = str(ancestor)[:3] # Get the first 3 characters of the ancestor code
            # lookup the ancestor code
            ancestor_title = icd9cm.lookup(ancestor)
            return ancestor_title
            # return icd9cm.lookup(icd_code)
        elif str(int(icd_version)) == '10':
            icd10cm = InnerMap.load("ICD10CM")
            ancestor = icd10cm.get_ancestors(icd_code)[1] # Get the closest ancestor
            ancestor = str(ancestor)[:3] # Get the first 3 characters of the

            return icd10cm.lookup(ancestor)
            # return icd10cm.lookup(icd_code)
        else:
            raise ValueError(f"Unsupported ICD version: {icd_version}")
    except KeyError:
        # Return None or a default value when code is not found
        return None
    except Exception as e:
        # Handle other potential errors
        print(f"Error looking up code {icd_code} for version {icd_version}: {e}")
        return None
    

def lookup_icd_code_static(icd_version, icd_code):
    """
    Lookup ICD code using InnerMap based on version with error handling.
    """
    icd_code = str(icd_code).strip()  # Ensure icd_code is a string and stripped of whitespace
    try:
        if str(int(icd_version)) == '9':
            icd9cm = InnerMap.load("ICD9CM")
            return icd9cm.lookup(icd_code)
        elif str(int(icd_version)) == '10':
            icd10cm = InnerMap.load("ICD10CM")
            return icd10cm.lookup(icd_code)
        else:
            raise ValueError(f"Unsupported ICD version: {icd_version}")
    except KeyError:
        # Return None or a default value when code is not found
        return None
    except Exception as e:
        # Handle other potential errors
        print(f"Error looking up code {icd_code}: {e}")
        return None
    

def load_data(data_path, filter=None, target_col='icd_code_broad'):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if filter:
            df = df[df[target_col].map(df[target_col].value_counts()) >= filter]

    return df

def drop_columns(df, columns):
    df = df.drop(columns=columns, errors='ignore')
    return df

def check_missing_values(df):
    missing = df.isnull().sum()
    return missing[missing > 0]


def plot_icd_code_distribution(df, output_path=None):


    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='icd_code_broad', order=df['icd_code_broad'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Samples per class distribution', fontsize=16)
    plt.tight_layout()
    if output_path is None:
        output_path = Path(FIG_PATH) / 'icd_code_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def convert_categorical_to_numerical(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    return df

def one_hot_encode(df, columns, target_col=None, staging=True):
    if target_col and target_col in columns:
        df[target_col] = df[target_col].astype('category').cat.codes

        target = df[target_col]



        if staging:
            df['disposition_grouped'] = df['disposition_grouped'].astype('category').cat.codes
            disposition  = df['disposition_grouped']
            if 'disposition_grouped' in df.columns and target_col != 'disposition_grouped':
                # If 'disposition_grouped' is present, drop it
                df = df.drop(columns=['disposition_grouped'])
        elif 'disposition_grouped' in df.columns and target_col != 'disposition_grouped' and staging is False:
            # If 'disposition_grouped' is present, drop it
            df = df.drop(columns=['disposition_grouped'])
        # drop the target column from the dataframe
        df = df.drop(columns=[target_col])

    for col in columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    if staging:
        df = pd.concat([df, disposition], axis=1)
    if target_col:
        # concatenate the target column back to the dataframe
        df = pd.concat([df, target], axis=1)
    return df


def create_train_test_split(df, target_col, test_size=0.2, random_state=42):
    Y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test



def append_constant_to_config(constant_name, constant_value, config_file_path='config.py'):
    """
    Appends a constant (variable name and its value) to a Python configuration file.
    If the constant with the given name already exists in the file, the function
    will skip appending and print a message indicating it's already present.

    This function serializes the constant_value using repr() to ensure it's
    written in a valid Python syntax, including proper handling of strings,
    lists, dictionaries, etc.

    Parameters:
    -----------
    constant_name : str
        The name of the constant (variable) to append (e.g., 'LOOK_UP', 'API_KEY').
    constant_value : any
        The value of the constant. This can be a string, number, list, dictionary,
        or any other Python object that can be represented by repr().
    config_file_path : str, optional
        The path to the configuration file to append to. Defaults to 'config.py'.

    Returns:
    --------
    bool
        True if the constant was successfully appended or was already present, False otherwise.
    """
    if not isinstance(constant_name, str) or not constant_name.isidentifier():
        print(f"Error: '{constant_name}' is not a valid Python identifier for a constant name.")
        return False

    try:
        # Check if the file exists and read its content
        file_content = ""
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as f:
                file_content = f.read()

        # Check if the constant_name is already present in the file
        # We look for the constant name followed by an equals sign, possibly with spaces
        # This is a simple check; for more robust parsing, a dedicated config parser would be better.
        if f"{constant_name} =" in file_content or f"{constant_name}=" in file_content:
            print(f"Constant '{constant_name}' already exists in {config_file_path}. Skipping append.")
            return True # Return True as the constant is effectively "there"

        # Serialize the constant_value using repr() to ensure proper string
        # representation for writing to a Python file.
        constant_value_repr = repr(constant_value)

        # Define the content to write
        content_to_append = f"\n{constant_name} = {constant_value_repr}\n"

        # Append to the config file
        with open(config_file_path, 'a') as f:
            f.write(content_to_append)

        print(f"Successfully appended '{constant_name}' to {config_file_path}")
        return True

    except FileNotFoundError:
        # If the file doesn't exist initially, it will be created by 'a' mode,
        # so this specific error might only occur if the directory is invalid.
        print(f"Error: Could not access or create the file '{config_file_path}'. Check path and permissions.")
        return False
    except IOError as e:
        print(f"Error writing to file '{config_file_path}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def create_diagnosis_column(df):
    """
    Creates a 'diagnosis' column based on the most prevalent disease category.

    Args:
        df (pd.DataFrame): The input DataFrame with disease categories as columns.
                          Assumes disease columns start from the 3rd column (index 2).

    Returns:
        pd.DataFrame: The DataFrame with the new 'diagnosis' column.
    """

    # Identify disease columns (assuming they start from the 3rd column, index 2)
    # Adjust this range if your actual disease columns are different.
    disease_columns = df.columns[2:15] # Columns 2 to 14 (inclusive of 14)

    # Initialize the 'diagnosis' column with a default value (e.g., None or 'Uncertain')
    df['diagnosis'] = None

    # Iterate through each row to determine the most prevalent diagnosis
    for index, row in df.iterrows():
        # Get the values for disease categories for the current row
        disease_values = row[disease_columns]

        # Filter out NaN values and -1.0 (uncertain)
        # We are looking for values that indicate a positive finding (e.g., 1.0)
        positive_findings = disease_values[disease_values == 1.0]

        if not positive_findings.empty:
            # If there are positive findings, find the most prevalent one.
            # In a multi-label scenario where multiple are 1.0, we need a tie-breaking rule.
            # For simplicity, we'll just pick the first one found if multiple are 1.0.
            # If a more complex prevalence rule is needed (e.g., based on external data),
            # that would need to be incorporated here.
            df.at[index, 'diagnosis'] = positive_findings.index[0]
        elif 'No Finding' in disease_columns and row['No Finding'] == 1.0:
            # If no positive findings but 'No Finding' is 1.0, assign 'No Finding'
            df.at[index, 'diagnosis'] = 'No Finding'
        else:
            # If no positive findings and 'No Finding' is not 1.0,
            # and there are uncertain values or all are NaN, label as 'Uncertain'
            # This covers cases where all are NaN or contain -1.0 without any 1.0
            df.at[index, 'diagnosis'] = 'Uncertain'

    return df

def calculate_mean_std(dataloader):
    # src https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch
    nimages = 0
    mean = 0.0
    var = 0.0

    for i_batch, batch_target in enumerate(tqdm(dataloader, desc="Calculating mean and std")):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print(f"Calculated mean: {mean}, std: {std}")
    return mean, std

def calculate_dataset_statistics(dataloader):
    """
    Calculates the mean and standard deviation of a PyTorch dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset.

    Returns:
        tuple: A tuple containing the mean and standard deviation.
    """
    # A single-channel image, so we'll collect sums for the single channel
    mean_sum = 0.0
    std_sum = 0.0
    total_pixels = 0
    
    print("Calculating dataset statistics. This may take a moment...")
    
    # Iterate through the dataset
    for images, _ in dataloader:
        # Images should be of shape (batch_size, channels, height, width)
        # We need to flatten the channel dimensions and get the pixel count
        batch_size = images.size(0)
        num_pixels_in_batch = images.size(2) * images.size(3)
        total_pixels += batch_size * num_pixels_in_batch

        # Sum up the pixel values for mean calculation
        mean_sum += torch.sum(images)
    
    # Calculate the mean
    mean = mean_sum / total_pixels
    print(f"Calculated mean: {mean.item():.4f}")

    # Reset total_pixels for the second pass (to avoid modifying the original variable)
    total_pixels_for_std = 0

    # Second pass to calculate the standard deviation
    for images, _ in dataloader:
        batch_size = images.size(0)
        num_pixels_in_batch = images.size(2) * images.size(3)
        total_pixels_for_std += batch_size * num_pixels_in_batch
        
        # Subtract the mean from each pixel and square it
        # Then sum it up for the variance calculation
        std_sum += torch.sum((images - mean)**2)
        
    # Calculate the standard deviation (sqrt of variance)
    std = torch.sqrt(std_sum / (total_pixels_for_std - 1))
    print(f"Calculated standard deviation: {std.item():.4f}")
    
    return mean.item(), std.item()


def stratified_subsample(df, label_col, frac=0.1, random_state=None):
    """
    Returns a stratified subsample of the DataFrame, keeping the class distribution
    proportional to the original dataset, with index reset.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    label_col : str
        Column name containing class labels.
    frac : float
        Fraction of the dataset to keep (0 < frac <= 1.0).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Stratified subsample with the same class proportions and reset index.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be between 0 and 1")

    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(frac=frac, random_state=random_state))
          .reset_index(drop=True)
    )



def one_hot_encode_eicu(df, columns, target_col=None, excluded_targets=None):
    if target_col and target_col in columns:
        df[target_col] = df[target_col].astype('category').cat.codes

        target = df[target_col]


        new_targets = [t for t in excluded_targets if t != target_col]

        df = df.drop(columns=new_targets)
        # drop the target column from the dataframe
        df = df.drop(columns=[target_col])

    for col in columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    if target_col:
        # concatenate the target column back to the dataframe
        df = pd.concat([df, target], axis=1)
    return df


def preprocess_eicu_data_advanced(
    output_path, 
    filename='eicu_merged_dataset.csv',
    filter_size=6000,
    target_col='mortality_risk_category',
    impute_missing=False,
    staging=True,
    sub_sample = None,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    verbose = False,
):
    """
    Complete preprocessing pipeline for MIMIC multimodal data.

    Parameters:
    -----------
    output_path : str
        Path to the directory containing the data file
    filename : str
        Name of the CSV file to load
    filter_size : int
        Number of rows to filter (passed to load_data)
    target_col : str
        Name of the target column
    impute_missing : bool
        Whether to impute missing values (True) or drop rows (False)
    sub_sample : float
        Whether to create a subset of the dataset for faster experimentation [0,1]
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of training data for validation set
    random_state : int
        Random state for reproducibility
    verbose : bool
        If True, log INFO messages; otherwise only WARNING and above.
    
    Returns:
    --------
    dict : Contains all processed data splits and metadata
    """
    if verbose:
        print("="*60)
        print("STARTING MIMIC DATA PREPROCESSING PIPELINE")
        print("="*60)
    
    # Step 1: Load data
    if verbose:
        print("Step 1: Loading data...")
    df_path = os.path.join(output_path, filename)
    df = load_data(df_path, filter=filter_size, target_col=target_col)
    if verbose:
        print("Loaded data shape: %s", df.shape)
    
    # Step 2: Drop unnecessary columns (excluding 'path' column)
    if verbose:
        print("\nStep 2: Dropping unnecessary columns...")
    columns_to_discard = [
        'patientunitstayid', 'admission_diagnoses', 'num_admission_diagnoses', 'admission_diagnoses','num_icd9_codes',
        'icd_code', 'primary_icd9_code','age','all_icd9_codes','sepsis_severity','cardiac_condition_category','primary_diagnosis_category'
    ]  # Removed 'path' from this list
    target_cols = ['mortality_risk_category', 'los_category', 'severity_category','discharge_category','resource_category']
    # Only drop columns that actually exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_discard if col in df.columns]
    missing_columns = [col for col in columns_to_discard if col not in df.columns]
    
    if existing_columns_to_drop:
        df = drop_columns(df, existing_columns_to_drop)
        if verbose:
            print("Dropped columns: %s", existing_columns_to_drop)
    
    if missing_columns and verbose:
        print("Columns not found (skipped): %s", missing_columns)
    if verbose:
        print("Data shape after dropping columns: %s", df.shape)
        print("Note: 'path' column retained for output data frames")
        
    # Step 3: Handle missing values
    if verbose:
        print("\nStep 3: Handling missing values...")
        print("Imputation strategy: %s", 'Impute' if impute_missing else 'Drop rows')
    
    initial_rows = len(df)
    df_original = df[target_col].copy()
    df = handle_missing_values(df, impute=impute_missing, target_col=target_col, verbose=verbose)
    final_rows = len(df)
    
    if not impute_missing and verbose:
        print("Rows before/after handling missing values: %d -> %d", initial_rows, final_rows)
    
    # Step 4: Check class distribution
    if verbose:
        print("\nStep 4: Analyzing target variable distribution...")
    class_distribution = df[target_col].value_counts()
    if verbose:
        print("Class distribution for '%s':", target_col)
    for class_name, count in class_distribution.items():
        percentage = (count / len(df)) * 100
        if verbose:
            print("  %s: %d (%.1f%%)", class_name, count, percentage)
    
    # Step 5: Feature normalization and encoding
    if verbose:
        print("\nStep 5: Feature preprocessing...")
    
    # Define columns to encode (categorical) - path is excluded from encoding
    columns_to_encode = ['gender', target_col, 'race', 'arrival_transport', 'pain','ethnicity']

    # Only include columns that exist in the dataframe
    existing_encode_cols = [col for col in columns_to_encode if col in df.columns]
    missing_encode_cols = [col for col in columns_to_encode if col not in df.columns]
    
    if missing_encode_cols and verbose:
        print("Encoding columns not found (skipped): %s", missing_encode_cols)
    
    # Normalize numerical columns (except target, categorical columns, and path)
    if verbose:
        print("Class count before subsampling\n%s", df[target_col].value_counts())
    if sub_sample is not None:
        # df = subsample_df(df, target_col=target_col, n_per_class=sub_sample)
        df = stratified_subsample(df, label_col=target_col, frac=sub_sample)
    if verbose:
        print("Normalizing numerical features...")
    df_normalized = df.copy()

    # Exclude 'path' from normalization
    exclude_from_normalization = existing_encode_cols
    
    for column in df.columns:
        if (column not in exclude_from_normalization and 
            df[column].dtype in ['int64', 'float64'] and 
            df[column].abs().max() != 0):
            
            df_normalized[column] = df[column] / df[column].abs().max()
    
    numerical_cols_normalized = [col for col in df.columns 
                                if col not in exclude_from_normalization 
                                and df[col].dtype in ['int64', 'float64']]
    if verbose:
        print("Normalized %d numerical columns", len(numerical_cols_normalized))
    
    # One-hot encode categorical columns (path is preserved as-is)
    if verbose:
        print("Applying one-hot encoding...")
    df_encoded = one_hot_encode_eicu(df_normalized, existing_encode_cols, target_col=target_col, excluded_targets=target_cols)
    if verbose:
        print("Data shape after encoding: %s", df_encoded.shape)
    # df_encoded['age'] = pd.to_numeric(df_encoded['age'], errors='coerce')

    try:
        # Step 6: Check before splitting
        check_min_samples_per_class(df_encoded[target_col], test_size=test_size, val_size=val_size)
    except ValueError as e:
        print("⚠️ Class imbalance issue detected: %s", e)
        print("Excluding rare classes and retrying...")

        df_encoded = exclude_rare_classes(df_encoded,target_col=target_col,test_size=test_size,val_size=val_size,min_per_class=100)

        # Re-check after exclusion
        check_min_samples_per_class(df_encoded[target_col], test_size=test_size, val_size=val_size)

    # Step 6: Create train/test split
    if verbose:
        print("\nStep 6: Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_encoded, target_col, test_size=test_size, random_state=random_state
    )
    if verbose:
        print("Train set: %s, Test set: %s", X_train.shape, X_test.shape)
    
    # Step 7: Scale numerical features (excluding path column)
    if verbose:
        print("\nStep 7: Scaling numerical features...")
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 'path' from numerical columns if it somehow got included
    if 'path' in numerical_columns:
        numerical_columns.remove('path')
    if verbose:
        print("Found %d numerical columns to scale", len(numerical_columns))
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numerical_columns:
        X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
        if verbose:
            print("Numerical features scaled using StandardScaler")
    
    # Step 8: Create validation split
    if verbose:
        print("\nStep 8: Creating validation split...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train
    )
    if verbose:
        print("Final splits:")
        print("Training: %s", X_train_final.shape)
        print("Validation: %s", X_val.shape)
        print("Test: %s", X_test_scaled.shape)
    
    # Verify path column is present in all splits
    path_in_train = 'path' in X_train_final.columns
    path_in_val = 'path' in X_val.columns
    path_in_test = 'path' in X_test_scaled.columns
    if verbose:
        print("\nPath column verification:")
        print("  In training set: %s", path_in_train)
        print("  In validation set: %s", path_in_val)
        print("  In test set: %s", path_in_test)
        
    # Summary
    if verbose:
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("="*60)
        print("Original data: %d rows", initial_rows)
        print("Final data: %d rows", len(df_encoded))
        print("Features: %d (including path column)", X_train_final.shape[1])
        print("Target classes: %d", len(class_distribution))
        
    # Return comprehensive results
    return {
        'X_train': X_train_final,
        'X_val': X_val,
        'X_test': X_test_scaled,
        'y_train': y_train_final,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'class_distribution': class_distribution,
        'feature_columns': X_train_final.columns.tolist(),
        'dropped_columns': existing_columns_to_drop,
        'encoded_columns': existing_encode_cols,
        'numerical_columns': numerical_columns,
        'original_shape': (initial_rows, df.shape[1] + len(existing_columns_to_drop)),
        'final_shape': df_encoded.shape,
        'final_df': df_encoded,
        'df_original': df_original
    }



def check_min_samples_per_class(y, test_size=0.2, val_size=0.2):
    """
    Ensure each class in y has at least enough samples to allow stratified splitting.
    """
    class_counts = y.value_counts()
    
    # Minimum needed for test + validation (rounded up)
    min_required = int(1/(1 - test_size) * (2 / (1 - val_size)))  # heuristic

    too_small = class_counts[class_counts < 3]  # stricter: at least 3 samples
    if not too_small.empty:
        raise ValueError(
            f"Some classes have too few samples for stratified split: {too_small.to_dict()}\n"
            f"Consider merging rare classes, filtering them out, or using non-stratified splitting."
        )
    return True

def exclude_rare_classes(df, target_col, test_size=0.2, val_size=0.2, min_per_class=3):
    """
    Exclude classes in target_col that don't have enough samples 
    to support stratified train/val/test splitting.
    """
    class_counts = df[target_col].value_counts()
    
    # Each class must have at least (min_per_class × number of splits)
    # For train/val/test → at least 3 × min_per_class
    min_required = min_per_class * 3
    
    rare_classes = class_counts[class_counts < min_required].index.tolist()
    if rare_classes:
        print(f"Excluding rare classes (too few samples for splitting): {rare_classes}")
        df = df[~df[target_col].isin(rare_classes)].copy()
    
    return df


if __name__ == "__main__":
    pass