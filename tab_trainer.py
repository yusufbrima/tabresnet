import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TabularClassifier import AdvancedTabularClassifier
from models.trainer import tabevaluate, tabtrain
from data.dataset import TabularDataset
from utils import preprocess_mimic_data_advanced, drop_columns, quantify_dataset_imbalance,\
    preprocess_eicu_data_advanced, get_class_weights,convert_to_serializable
import os
import pandas as pd
from config import OUTPUT_PATH, TABULAR_EPOCHS, FILTER_SIZE, VAL_SIZE, TEST_SIZE, TARGET_COL, \
    PATIENCE, RESULT_PATH, CUTOFF, EICU_TARGETS, EICU_PATH, EICU_FILE, DATASET_FLAG, EXPERIMENT_ID,RANDOM_SEED
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import time
import argparse

# --------------------------
# GLOBAL EXPERIMENT ID
# --------------------------
experiment_id = EXPERIMENT_ID  # change this for each new experiment

# Start timing the overall experiment
overall_start_time = time.time()

dataset_flag = DATASET_FLAG # "mimic" or "eicu"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run models with specified target column.')
parser.add_argument('--target_col', type=str, default=TARGET_COL if dataset_flag=='mimic' else EICU_TARGETS[0],
                    help='Target column to use (default from config)')
parser.add_argument('--weighting_strategy', type=str, default="inverse",
                    choices=["inverse", "effective", "median", "noweighting"],
                    help="Class weighting strategy: 'inverse', 'effective', or 'median'")

args = parser.parse_args()
target_col = args.target_col
weighting_strategy = args.weighting_strategy

# Ensure FILTER_SIZE is a list (reverse order for descending sizes)
# FILTER_SIZE = [500]
# dataset_flag = 'mimic'
FILTER_SIZE.reverse()
filter_sizes = FILTER_SIZE

print(f"Running experiments with filter sizes: {filter_sizes}")

# Prepare a results container
all_results = {}

for filter_size in filter_sizes:
    print(f"\n\n===== Running experiment for FILTER_SIZE = {filter_size} =====")
    if dataset_flag == 'mimic':
        print("Processing MIMIC dataset...")
        processed_data = preprocess_mimic_data_advanced(
            output_path=OUTPUT_PATH,
            filename=CUTOFF if CUTOFF else 'mimic_multimodal_image_centric_streamlined_found.csv',
            filter_size=filter_size,
            target_col=target_col,
            impute_missing=True,
            staging=False,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_SEED
        )

        X_train_part = processed_data['X_train']
        X_val = processed_data['X_val']
        X_test = processed_data['X_test']
        y_train_part = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']

        # Drop irrelevant columns
        if target_col == 'diagnosis':
            cols_to_drop = ['icd_code_broad', 'disposition_grouped']
        elif target_col == 'disposition_grouped':
            cols_to_drop = ['icd_code_broad', 'diagnosis']
        elif target_col == 'icd_code_broad':
            cols_to_drop = ['diagnosis', 'disposition_grouped']
        for df in [X_train_part, X_val, X_test]:
            df.drop(columns=cols_to_drop + ['path'], inplace=True, errors='ignore')
    else:
        print("Processing eICU dataset...")
        processed_data = preprocess_eicu_data_advanced(
            output_path=OUTPUT_PATH,
            filename=EICU_FILE,
            target_col=target_col,
            impute_missing=True,
            filter_size=filter_size,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=42
        )
        X_train_part = processed_data['X_train'].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        X_val = processed_data['X_val'].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        X_test = processed_data['X_test'].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        y_train_part = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']

    # Compute class weights
    class_weights, unique_classes = get_class_weights(y_train_part, strategy=weighting_strategy, beta=0.9999)
    class_counts = [np.sum(y_train_part == cls) for cls in unique_classes]

    print("Class weights:", class_weights)
    imbalance_metrics = quantify_dataset_imbalance(class_counts=class_counts, class_weights=class_weights)
    print(f"Imbalance metrics ({weighting_strategy}): {imbalance_metrics}")

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Print class info
    class_info = {}
    for i, weight in enumerate(class_weights):
        class_count = np.sum(y_train_part == i)
        class_info[f"class_{i}"] = {
            "weight": float(weight),
            "count": int(class_count),
            "percentage": float(class_count / len(y_train_part) * 100)
        }
        print(f"Class {i}: weight = {weight:.4f}, count = {class_count}")

    # Create datasets and loaders
    train_dataset = TabularDataset(X_train_part.to_numpy(dtype=np.float32), y_train_part)
    val_dataset = TabularDataset(X_val.to_numpy(dtype=np.float32), y_val)
    test_dataset = TabularDataset(X_test.to_numpy(dtype=np.float32), y_test)

    batch_size = 2048
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = X_train_part.shape[1]
    output_dim = len(np.unique(y_train_part))
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    models = {
        "AdvancedTabularClassifier": AdvancedTabularClassifier(input_dim, output_dim, hidden_dim=520, num_blocks=4, use_reduction=False, dropout=0.0)
    }

    for name, model in models.items():
        print(f"\n===== Training {name} with FILTER_SIZE={filter_size} =====")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights_tensor = class_weights.to(device)

        if weighting_strategy == 'noweighting':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0009091928042586428, weight_decay=1.5617683600911463e-05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        model.to(device)
        print(f"Using device: {device}")

        num_epochs = TABULAR_EPOCHS
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        learning_rates = []

        print("Starting training...")
        train_start = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            train_loss, train_acc = tabtrain(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = tabevaluate(model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            epoch_time = time.time() - epoch_start_time

            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
                  f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f'best_model_{name}_filter_{filter_size}_{experiment_id}.pth')
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        train_end = time.time()
        training_time = train_end - train_start

        # Load best model
        model.load_state_dict(torch.load(f'best_model_{name}_filter_{filter_size}_{experiment_id}.pth'))

        # Evaluate on test set
        test_loss, test_acc, test_preds, test_labels = tabevaluate(model, test_loader, criterion, device)

        # Metrics
        test_f1_macro = f1_score(test_labels, test_preds, average='macro')
        test_f1_micro = f1_score(test_labels, test_preds, average='micro')
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        test_precision_macro = precision_score(test_labels, test_preds, average='macro')
        test_recall_macro = recall_score(test_labels, test_preds, average='macro')
        conf_matrix = confusion_matrix(test_labels, test_preds)
        class_report_dict = classification_report(test_labels, test_preds, output_dict=True)

        # Initialize dictionary for the model if not exist
        if name not in all_results:
            all_results[name] = {
                "filter_sizes": [],
                "training_time_seconds": [],
                "num_classes": [],
                "test_f1_macro": [],
                "test_f1_micro": [],
                "test_f1_weighted": [],
                "cv_class_weights": [],
                "imbalance_ratio": [],
                "total_training_samples": [],
                "entropy": [],
                "test_precision_macro": [],
                "test_recall_macro": [],
                "classification_reports": [],
                "confusion_matrices": []
            }

        # Append current filter size results to model dictionary
        all_results[name]["filter_sizes"].append(filter_size)
        all_results[name]["training_time_seconds"].append(training_time)
        all_results[name]["num_classes"].append(len(unique_classes))
        all_results[name]["test_f1_macro"].append(test_f1_macro)
        all_results[name]["test_f1_micro"].append(test_f1_micro)
        all_results[name]["test_f1_weighted"].append(test_f1_weighted)
        all_results[name]["cv_class_weights"].append(imbalance_metrics["cv_class_weights"])
        all_results[name]["imbalance_ratio"].append(imbalance_metrics["imbalance_ratio"])
        all_results[name]["total_training_samples"].append(len(X_train_part))
        all_results[name]["entropy"].append(imbalance_metrics["entropy"])
        all_results[name]["test_precision_macro"].append(test_precision_macro)
        all_results[name]["test_recall_macro"].append(test_recall_macro)
        all_results[name]["classification_reports"].append(class_report_dict)
        all_results[name]["confusion_matrices"].append(conf_matrix.tolist())

        print(f"Metrics for FILTER_SIZE={filter_size} added to results for model {name}")

        print(f"\nFinal Test Results for filter_size={filter_size}:")
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        print(f"Test F1 (macro): {test_f1_macro:.4f}")
        print(f"Test F1 (micro): {test_f1_micro:.4f}")
        print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds))

# Save all results to a single JSON file
json_filename = f"all_neural_models_metrics_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json"
json_path = os.path.join(RESULT_PATH, json_filename)

with open(json_path, "w") as f:
    json.dump(convert_to_serializable(all_results), f, indent=4)


print(f"\nAll metrics saved to {json_path}")

# End overall experiment timer
overall_end_time = time.time()
print(f"\nTotal runtime (all filter sizes and models): {(overall_end_time - overall_start_time) / 60:.2f} minutes")