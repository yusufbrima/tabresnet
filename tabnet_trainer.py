import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from models.trainer import tabevaluate, tabtrain
from data.dataset import TabularDataset
from utils.helper import preprocess_mimic_data_advanced, drop_columns, quantify_dataset_imbalance,\
    preprocess_eicu_data_advanced,get_class_weights,convert_to_serializable
import os 
from pytorch_tabnet.tab_model import TabNetClassifier
from config import OUTPUT_PATH, TABULAR_EPOCHS, FILTER_SIZE, VAL_SIZE, TEST_SIZE, TARGET_COL, \
    PATIENCE,RESULT_PATH,CUTOFF,EICU_TARGETS, EICU_PATH, EICU_FILE,MIMIC_TARGETS, DATASET_FLAG
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import time
import argparse


# Start timing the overall experiment
overall_start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: parse dataset_flag first
parser = argparse.ArgumentParser(description="Run models with specified configuration.")
parser.add_argument("--dataset_flag", type=str, default="eicu", choices=["eicu", "mimic"], help="Dataset flag (either 'eicu' or 'mimic').")
args, remaining_argv = parser.parse_known_args()
dataset_flag = args.dataset_flag

# Step 2: create full parser now that dataset_flag is known
parser = argparse.ArgumentParser(description="Run models with specified configuration.")
parser.add_argument("--target_col", type=str, default=TARGET_COL if dataset_flag == "mimic" else EICU_TARGETS[0], help="Target column to use (default from config).")
parser.add_argument("--weighting_strategy", type=str, default="inverse", choices=["inverse", "effective", "median", "noweighting"], help="Class weighting strategy.")
parser.add_argument("--experiment_id", type=int, default=50, help="Experiment ID (integer, default=1).")
parser.add_argument("--dataset_flag", type=str, default=dataset_flag, choices=["eicu", "mimic"], help="Dataset flag (either 'eicu' or 'mimic').")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")

args = parser.parse_args(remaining_argv)

# Assign arguments to variables
experiment_id = args.experiment_id
dataset_flag = args.dataset_flag
target_col = args.target_col
weighting_strategy = args.weighting_strategy
random_seed = args.random_seed



if dataset_flag == 'eicu':
    # Filter Sizes for EICU
    FILTER_SIZE = [300, 1000, 1500, 3500, 4500, 6000, 8000, 12000]
else:
    # Filter Sizes for MIMIC-IV 
    FILTER_SIZE = [500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250, 3500, 3750, 4000]


FILTER_SIZE.reverse()
filter_sizes = FILTER_SIZE

print(f"Running experiments with filter sizes: {filter_sizes}")

# Prepare a results container
all_results = {}
all_results_json_filename = os.path.join(RESULT_PATH, f"all_results_tabnet_results_filter_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json")

for filter_size in filter_sizes:
    print(f"\n\n===== Running experiment for FILTER_SIZE = {filter_size} =====")
    
    # Start timing the experiment
    start_time = time.time()
    
    if dataset_flag == 'mimic':
        processed_data = preprocess_mimic_data_advanced(
            output_path=OUTPUT_PATH,
            filename= CUTOFF if CUTOFF else 'mimic_multimodal_image_centric_streamlined_found.csv',
            filter_size=filter_size,
            staging=False,
            target_col=target_col,
            impute_missing=True,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=random_seed
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
        processed_data = preprocess_eicu_data_advanced(
            output_path=OUTPUT_PATH,
            filename=EICU_FILE,
            target_col=target_col,
            impute_missing=True,
            filter_size=filter_size,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=random_seed
        )
        X_train_part = processed_data['X_train']
        X_val = processed_data['X_val']
        X_test = processed_data['X_test']
        y_train_part = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']

    # Compute class weights
    class_weights, unique_classes = get_class_weights(y_train_part, strategy=weighting_strategy, beta=0.9999)
    class_counts = [np.sum(y_train_part == cls) for cls in unique_classes]
    imbalance_metrics = quantify_dataset_imbalance(class_counts=class_counts, class_weights=class_weights)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Class info dictionary
    class_info = {}
    for i, weight in enumerate(class_weights):
        class_count = np.sum(y_train_part == i)
        class_info[f"class_{i}"] = {
            "weight": float(weight),
            "count": int(class_count),
            "percentage": float(class_count / len(y_train_part) * 100)
        }

    # Create datasets and loaders
    train_dataset = TabularDataset(X_train_part.to_numpy(dtype=np.float32), y_train_part)
    val_dataset = TabularDataset(X_val.to_numpy(dtype=np.float32), y_val)
    test_dataset = TabularDataset(X_test.to_numpy(dtype=np.float32), y_test)
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = X_train_part.shape[1]
    output_dim = len(np.unique(y_train_part))
    experiment_config = {
        "model_type": "TabPFNClassifier",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "batch_size": batch_size,
        "filter_size": filter_size,
        "dataset_info": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "num_features": input_dim,
            "num_classes": output_dim
        }
    }

    sample_weights = np.array([class_weights[y] for y in y_train_part])
    class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)}
    lr = 0.0018294599871120757
    weight_decay = 0.0017793935803061614
    models = {"TabNet": TabNetClassifier(optimizer_fn=torch.optim.AdamW,optimizer_params=dict(lr=lr, weight_decay=weight_decay),device_name=device.type)}

    X_train_part = X_train_part.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    y_train_part = pd.to_numeric(y_train_part, errors="coerce").astype(np.int64)
    y_val = pd.to_numeric(y_val, errors="coerce").astype(np.int64)

    for name, model in models.items():
        print(f"\n===== Training {name} with FILTER_SIZE={filter_size} =====")
        train_start = time.time()
        # model._default_loss = torch.nn.functional.cross_entropy
        if weighting_strategy == 'noweighting':
            model.fit(X_train_part.to_numpy(), y_train_part.to_numpy(), eval_set=[(X_val.to_numpy(), y_val.to_numpy())], batch_size=batch_size, patience=PATIENCE)
        else:
            model.fit(X_train_part.to_numpy(), y_train_part.to_numpy(), eval_set=[(X_val.to_numpy(), y_val.to_numpy())],weights=class_weight_dict, batch_size=batch_size, patience=PATIENCE)
        train_end = time.time()
        training_time = train_end - train_start

        test_preds = model.predict(X_test.to_numpy())

        # Metrics
        test_f1_macro = f1_score(y_test, test_preds, average='macro')
        test_f1_micro = f1_score(y_test, test_preds, average='micro')
        test_f1_weighted = f1_score(y_test, test_preds, average='weighted')
        test_precision_macro = precision_score(y_test, test_preds, average='macro')
        test_recall_macro = recall_score(y_test, test_preds, average='macro')
        conf_matrix = confusion_matrix(y_test, test_preds)
        class_report_dict = classification_report(y_test, test_preds, output_dict=True)

        if name not in all_results:
            all_results[name] = {
                "filter_sizes": [],
                "training_time_seconds": [],
                "num_classes": [],
                "test_f1_macro": [],
                "test_f1_micro": [],
                "test_f1_weighted": [],
                "cv_class_weights": [],
                "total_training_samples":[],
                "imbalance_ratio": [],
                "entropy": [],
                "test_precision_macro": [],
                "test_recall_macro": [],
                "classification_reports": [],
                "confusion_matrices": []
            }

        all_results[name]["filter_sizes"].append(filter_size)
        all_results[name]["training_time_seconds"].append(training_time)
        all_results[name]["num_classes"].append(len(unique_classes))
        all_results[name]["test_f1_macro"].append(test_f1_macro)
        all_results[name]["test_f1_micro"].append(test_f1_micro)
        all_results[name]["test_f1_weighted"].append(test_f1_weighted)
        all_results[name]["total_training_samples"].append(len(X_train_part))
        all_results[name]["cv_class_weights"].append(imbalance_metrics["cv_class_weights"])
        all_results[name]["imbalance_ratio"].append(imbalance_metrics["imbalance_ratio"])
        all_results[name]["entropy"].append(imbalance_metrics["entropy"])
        all_results[name]["test_precision_macro"].append(test_precision_macro)
        all_results[name]["test_recall_macro"].append(test_recall_macro)
        all_results[name]["classification_reports"].append(class_report_dict)
        all_results[name]["confusion_matrices"].append(conf_matrix.tolist())

    # Save results per filter
    with open(all_results_json_filename, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=4)

# Save all results to a single JSON file using experiment_id
json_filename = f"all_single_tabnet_models_metrics_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json"
json_path = os.path.join(RESULT_PATH, json_filename)
with open(json_path, "w") as f:
    json.dump(convert_to_serializable(all_results), f, indent=4)

print(f"\nAll metrics saved to {json_path}")

# End overall experiment timer
overall_end_time = time.time()
print(f"\nTotal runtime (all filter sizes and models): {(overall_end_time - overall_start_time) / 60:.2f} minutes")
