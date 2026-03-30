import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tabpfn import TabPFNClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import argparse
import os
import time
import json

from utils.helper import preprocess_mimic_data_advanced, preprocess_eicu_data_advanced, get_class_weights, quantify_dataset_imbalance, convert_to_serializable
from data.dataset import TabularDataset
from config import OUTPUT_PATH, VAL_SIZE, TEST_SIZE, TARGET_COL, RESULT_PATH, CUTOFF, EICU_TARGETS, EICU_FILE

# Start timing
overall_start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------
# Argument parsing
# ---------------------
parser = argparse.ArgumentParser(description="Run TabPFN experiments.")
parser.add_argument("--dataset_flag", type=str, default="eicu", choices=["eicu", "mimic"])
args, remaining_argv = parser.parse_known_args()
dataset_flag = args.dataset_flag

parser = argparse.ArgumentParser(description="Run TabPFN experiments.")
parser.add_argument("--target_col", type=str, default=TARGET_COL if dataset_flag == "mimic" else EICU_TARGETS[0])
parser.add_argument("--weighting_strategy", type=str, default="inverse", choices=["inverse", "effective", "median", "noweighting"])
parser.add_argument("--experiment_id", type=int, default=50)
parser.add_argument("--dataset_flag", type=str, default=dataset_flag)
parser.add_argument("--random_seed", type=int, default=42)
args = parser.parse_args(remaining_argv)

experiment_id = args.experiment_id
dataset_flag = args.dataset_flag
target_col = args.target_col
weighting_strategy = args.weighting_strategy
RANDOM_SEED = args.random_seed

# ---------------------
# Filter sizes
# ---------------------
FILTER_SIZE = [300, 1000, 1500, 3500, 4500, 6000, 8000, 12000] if dataset_flag == 'eicu' else \
              [500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000]
FILTER_SIZE.reverse()

print(f"Running experiments with filter sizes: {FILTER_SIZE}")

all_results = {}

for filter_size in FILTER_SIZE:
    print(f"\n===== Running experiment for FILTER_SIZE = {filter_size} =====")

    # ---------------------
    # Dataset processing
    # ---------------------
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
        X_train = processed_data['X_train']
        X_val = processed_data['X_val']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']

        # Drop irrelevant columns
        cols_to_drop = {
            'diagnosis': ['icd_code_broad', 'disposition_grouped'],
            'disposition_grouped': ['icd_code_broad', 'diagnosis'],
            'icd_code_broad': ['diagnosis', 'disposition_grouped']
        }.get(target_col, [])
        for df in [X_train, X_val, X_test]:
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
            random_state=RANDOM_SEED
        )
        X_train = processed_data['X_train'].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        X_val = processed_data['X_val'].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        X_test = processed_data['X_test'].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        y_train = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']

    # ---------------------
    # Class weights & imbalance
    # ---------------------
    class_weights, unique_classes = get_class_weights(y_train, strategy=weighting_strategy, beta=0.9999)
    class_counts = [np.sum(y_train == cls) for cls in unique_classes]

    print("Class weights:", class_weights)
    imbalance_metrics = quantify_dataset_imbalance(class_counts=class_counts, class_weights=class_weights)
    print(f"Imbalance metrics ({weighting_strategy}): {imbalance_metrics}")

    # ---------------------
    # TabPFN training & evaluation
    # ---------------------
    print(f"\nTraining TabPFNClassifier for FILTER_SIZE={filter_size} ...")
    start_time = time.time()
    # src: https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/many_class/many_class_classifier_example.py
    clf = TabPFNClassifier(balance_probabilities=True,ignore_pretraining_limits=True,n_preprocessing_jobs=4)  # defaults to v2.5
    clf.to(device)
    clf.fit(X_train.to_numpy(), y_train.to_numpy())
    training_time = time.time() - start_time

    # Predictions
    y_pred = clf.predict(X_test.to_numpy())

    # Metrics
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_micro = f1_score(y_test, y_pred, average='micro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    test_precision_macro = precision_score(y_test, y_pred, average='macro')
    test_recall_macro = recall_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Store results
    all_results.setdefault("TabPFNClassifier", {
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
    })

    res = all_results["TabPFNClassifier"]
    res["filter_sizes"].append(filter_size)
    res["training_time_seconds"].append(training_time)
    res["num_classes"].append(len(unique_classes))
    res["test_f1_macro"].append(test_f1_macro)
    res["test_f1_micro"].append(test_f1_micro)
    res["test_f1_weighted"].append(test_f1_weighted)
    res["cv_class_weights"].append(imbalance_metrics["cv_class_weights"])
    res["imbalance_ratio"].append(imbalance_metrics["imbalance_ratio"])
    res["total_training_samples"].append(len(X_train))
    res["entropy"].append(imbalance_metrics["entropy"])
    res["test_precision_macro"].append(test_precision_macro)
    res["test_recall_macro"].append(test_recall_macro)
    res["classification_reports"].append(class_report_dict)
    res["confusion_matrices"].append(conf_matrix.tolist())

    print(f"Filter size {filter_size} done. F1 macro: {test_f1_macro:.4f}")

# ---------------------
# Save results
# ---------------------
json_filename = f"all_tabpfn_metrics_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json"
json_path = os.path.join(RESULT_PATH, json_filename)
with open(json_path, "w") as f:
    json.dump(convert_to_serializable(all_results), f, indent=4)

print(f"\nAll metrics saved to {json_path}")
print(f"\nTotal runtime: {(time.time() - overall_start_time)/60:.2f} minutes")
