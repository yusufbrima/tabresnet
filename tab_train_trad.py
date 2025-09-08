import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from utils import preprocess_mimic_data_advanced, drop_columns, quantify_dataset_imbalance,\
    preprocess_eicu_data_advanced, get_class_weights, convert_to_serializable
import os
from config import OUTPUT_PATH, FILTER_SIZE, VAL_SIZE, TEST_SIZE, TARGET_COL, RESULT_PATH, CUTOFF,\
    EICU_TARGETS, EICU_FILE,MIMIC_TARGETS, DATASET_FLAG, EXPERIMENT_ID, RANDOM_SEED
import time
import json
import argparse

# --------------------------
# GLOBAL EXPERIMENT ID
# --------------------------
experiment_id = EXPERIMENT_ID # change this for each new experiment

# Start timing the overall experiment
overall_start_time = time.time()

dataset_flag = DATASET_FLAG  # "mimic" "eicu"

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

# Sanitize feature names
def sanitize_column_names(df):
    df.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').strip() for c in df.columns]
    return df

# Reverse filter sizes for descending order
FILTER_SIZE.reverse()
filter_sizes = FILTER_SIZE
print(f"Running experiments with filter sizes: {filter_sizes}")

# Prepare results container
all_results = {}

for filter_size in filter_sizes:
    print(f"\n\n===== Running experiment for FILTER_SIZE = {filter_size} =====")
    
    # Load data
    if dataset_flag == 'mimic':
        processed_data = preprocess_mimic_data_advanced(
            output_path=OUTPUT_PATH,
            filename=CUTOFF if CUTOFF else 'mimic_multimodal_image_centric_streamlined_found.csv',
            filter_size=filter_size,
            target_col=target_col,
            staging=False,
            impute_missing=True,
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
        processed_data = preprocess_eicu_data_advanced(
            output_path=OUTPUT_PATH,
            filename=EICU_FILE,
            target_col=target_col,
            filter_size=filter_size,
            impute_missing=True,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=42
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

    # Sanitize feature names
    X_train_part = sanitize_column_names(X_train_part)
    X_val = sanitize_column_names(X_val)
    X_test = sanitize_column_names(X_test)

    # Convert features to numeric
    X_train_part = X_train_part.apply(pd.to_numeric, errors='coerce')
    X_val = X_val.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Define XGBoost params

    params = {
        'objective': 'multi:softprob' if len(unique_classes) > 2 else 'binary:logistic',
        'num_class': len(unique_classes) if len(unique_classes) > 2 else None,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'max_depth': 12,
        'min_child_weight': 8,
        'subsample': 0.8394633936788146,
        'colsample_bytree': 0.5780093202212182,
        'colsample_bylevel': 0.5779972601681014,
        'reg_alpha': 0.2904180608409973,
        'reg_lambda': 4.330880728874676,
        'random_state': RANDOM_SEED,
        'n_jobs': 4,
        'verbosity': 0,
        'eval_metric': 'mlogloss' if len(unique_classes) > 2 else 'logloss'
    }
   
    class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)}
    # Initialize models

    models = {
        "DecisionTree": DecisionTreeClassifier(class_weight=class_weight_dict if weighting_strategy != 'noweighting' else None, random_state=RANDOM_SEED,min_samples_split = 2, min_samples_leaf = 1, criterion = 'gini', max_depth = None),
        "RandomForest": RandomForestClassifier(n_estimators=331,max_depth=25,min_samples_split=50,min_impurity_decrease= 2.04713024607963e-06, bootstrap=True,class_weight=class_weight_dict if weighting_strategy != 'noweighting' else None, random_state=RANDOM_SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(**params)
    }

    # Train and evaluate
    for name, model in models.items():
        print(f"\n===== Training {name} with FILTER_SIZE={filter_size} =====")
        # print("Class weight ", class_weights)
        train_start = time.time()
        if name == "XGBoost":
            if weighting_strategy == 'noweighting':
                model.fit(X_train_part.values, y_train_part.values)
            else:
                sample_weights = np.array([class_weights[y] for y in y_train_part])
                model.fit(X_train_part.values, y_train_part.values, sample_weight=sample_weights)
            test_preds = model.predict(X_test.values)
        else:
            model.fit(X_train_part, y_train_part)
            test_preds = model.predict(X_test)
        train_end = time.time()
        training_time = train_end - train_start

        # Compute metrics
        test_f1_macro = f1_score(y_test, test_preds, average='macro')
        test_f1_micro = f1_score(y_test, test_preds, average='micro')
        test_f1_weighted = f1_score(y_test, test_preds, average='weighted')
        test_precision_macro = precision_score(y_test, test_preds, average='macro')
        test_recall_macro = recall_score(y_test, test_preds, average='macro')
        conf_matrix = confusion_matrix(y_test, test_preds)
        class_report_dict = classification_report(y_test, test_preds, output_dict=True)

        # Initialize model results
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

        # Append results
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

# Save all results using experiment_id
json_filename = f"all_models_metrics_traditional_models_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json"
json_path = os.path.join(RESULT_PATH, json_filename)
with open(json_path, "w") as f:
    json.dump(convert_to_serializable(all_results), f, indent=4)

print(f"\nAll metrics saved to {json_path}")

# End overall experiment timer
overall_end_time = time.time()
print(f"\nTotal runtime (all filter sizes and models): {(overall_end_time - overall_start_time) / 60:.2f} minutes")