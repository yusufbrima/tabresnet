"""
Improved Optuna hyperparameter search + final training/evaluation using ONLY the MIMIC dataset.
Usage example:
    python run_optuna_trad_mimic.py --target_col icd_code_broad 
"""

import numpy as np
import pandas as pd
import os
import json
import time
import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import preprocess_mimic_data_advanced, get_class_weights, quantify_dataset_imbalance, convert_to_serializable
from config import OUTPUT_PATH, RESULT_PATH, VAL_SIZE, TEST_SIZE, TARGET_COL, EXPERIMENT_ID

# --------------------------
# GLOBAL EXPERIMENT ID
# --------------------------
experiment_id = EXPERIMENT_ID

# Start timing the overall experiment
overall_start_time = time.time()

# Fixed filter size
FILTER_SIZE = 500
dataset_flag = "mimic"
n_trials = 100  # Increased from 20

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run models with specified target column.')
parser.add_argument('--target_col', type=str, default=TARGET_COL[0],
                    help='Target column to use (default from config)')
parser.add_argument('--weighting_strategy', type=str, default="inverse",
                    choices=["inverse", "effective", "median", "noweighting"],
                    help="Class weighting strategy: 'inverse', 'effective', or 'median'")
parser.add_argument('--use_cv', action='store_true',
                    help="Use cross-validation instead of validation set for hyperparameter tuning")
args = parser.parse_args()
target_col = args.target_col
weighting_strategy = args.weighting_strategy
use_cv = args.use_cv

# Sanitize feature names
def sanitize_column_names(df):
    df.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').strip() for c in df.columns]
    return df

# Load MIMIC data
processed_data = preprocess_mimic_data_advanced(
    output_path=OUTPUT_PATH,
    filename='mimic_multimodal_image_centric_streamlined_found.csv',
    filter_size=FILTER_SIZE,
    target_col=target_col,
    staging=False,
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

# Drop irrelevant columns
if target_col == 'diagnosis':
    cols_to_drop = ['icd_code_broad', 'disposition_grouped']
elif target_col == 'disposition_grouped':
    cols_to_drop = ['icd_code_broad', 'diagnosis']
elif target_col == 'icd_code_broad':
    cols_to_drop = ['diagnosis', 'disposition_grouped']
for df in [X_train_part, X_val, X_test]:
    df.drop(columns=cols_to_drop + ['path'], inplace=True, errors='ignore')

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

# --------------------------
# TEST DEFAULT BASELINES
# --------------------------
def test_default_baselines():
    """Test default parameters to establish baselines"""
    print("\n=== TESTING DEFAULT BASELINES ===")
    
    # Default Decision Tree
    class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)} if weighting_strategy != 'noweighting' else None
    dt_default = DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42)
    dt_default.fit(X_train_part, y_train_part)
    dt_preds = dt_default.predict(X_val)
    dt_f1 = f1_score(y_val, dt_preds, average='macro')
    print(f"Default DecisionTree F1: {dt_f1:.4f}")
    
    # Default Random Forest
    rf_default = RandomForestClassifier(n_estimators=200, class_weight=class_weight_dict, random_state=42, n_jobs=-1)
    rf_default.fit(X_train_part, y_train_part)
    rf_preds = rf_default.predict(X_val)
    rf_f1 = f1_score(y_val, rf_preds, average='macro')
    print(f"Default RandomForest F1: {rf_f1:.4f}")
    
    # Default XGBoost (your good parameters)
    xgb_params = {
        'objective': 'multi:softprob' if len(unique_classes) > 2 else 'binary:logistic',
        'num_class': len(unique_classes) if len(unique_classes) > 2 else None,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': 4,
        'verbosity': 0
    }
    xgb_default = XGBClassifier(**xgb_params)
    if weighting_strategy != 'noweighting':
        sample_weights = np.array([class_weights[y] for y in y_train_part])
        xgb_default.fit(X_train_part.values, y_train_part.values, sample_weight=sample_weights)
    else:
        xgb_default.fit(X_train_part.values, y_train_part.values)
    xgb_preds = xgb_default.predict(X_val.values)
    xgb_f1 = f1_score(y_val, xgb_preds, average='macro')
    print(f"Default XGBoost F1: {xgb_f1:.4f}")
    
    print("Optuna must beat these scores!\n")
    return {"DecisionTree": dt_f1, "RandomForest": rf_f1, "XGBoost": xgb_f1}

# Test baselines
baseline_scores = test_default_baselines()

# --------------------------
# IMPROVED OPTUNA OBJECTIVES
# --------------------------

def objective_decision_tree(trial):
    """Conservative Decision Tree optimization"""
    # Allow unlimited depth (sklearn default) as an option
    use_unlimited_depth = trial.suggest_categorical("use_unlimited_depth", [True, False])
    
    if use_unlimited_depth:
        max_depth = None
    else:
        max_depth = trial.suggest_int("max_depth", 3, 20)
    
    params = {
        "max_depth": max_depth,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
    
    # Optionally add more complex parameters
    if trial.suggest_categorical("use_advanced_params", [True, False]):
        params["max_features"] = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
        params["min_impurity_decrease"] = trial.suggest_float("min_impurity_decrease", 0.0, 0.01)
    
    class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)} if weighting_strategy != 'noweighting' else None
    model = DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42, **params)
    
    if use_cv:
        # Use cross-validation for more robust estimates
        cv_scores = cross_val_score(model, X_train_part, y_train_part, cv=3, scoring='f1_macro', n_jobs=-1)
        return cv_scores.mean()
    else:
        model.fit(X_train_part, y_train_part)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average='macro')

def objective_random_forest(trial):
    """Improved Random Forest optimization"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }
    
    # If bootstrap=False, we can't use oob_score
    # if not params["bootstrap"]:
    #     params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
    
    class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)} if weighting_strategy != 'noweighting' else None
    model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42, n_jobs=-1, **params)
    
    if use_cv:
        cv_scores = cross_val_score(model, X_train_part, y_train_part, cv=3, scoring='f1_macro', n_jobs=-1)
        return cv_scores.mean()
    else:
        model.fit(X_train_part, y_train_part)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average='macro')

def objective_xgboost(trial):
    """Improved XGBoost optimization with early stopping"""
    params = {
        "objective": "multi:softprob" if len(unique_classes) > 2 else "binary:logistic",
        "num_class": len(unique_classes) if len(unique_classes) > 2 else None,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200) if use_cv else 1000,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": 0
    }
    
    model = XGBClassifier(**params)
    
    if use_cv:
        # For CV, we can't use early stopping easily, so use the suggested n_estimators
        if weighting_strategy != 'noweighting':
            sample_weights = np.array([class_weights[y] for y in y_train_part])
            # Custom CV with sample weights
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train_part, y_train_part):
                X_cv_train, X_cv_val = X_train_part.iloc[train_idx], X_train_part.iloc[val_idx]
                y_cv_train, y_cv_val = y_train_part.iloc[train_idx], y_train_part.iloc[val_idx]
                cv_sample_weights = sample_weights[train_idx]
                
                model.fit(X_cv_train.values, y_cv_train.values, sample_weight=cv_sample_weights)
                cv_preds = model.predict(X_cv_val.values)
                cv_scores.append(f1_score(y_cv_val, cv_preds, average='macro'))
            return np.mean(cv_scores)
        else:
            cv_scores = cross_val_score(model, X_train_part.values, y_train_part.values, 
                                      cv=3, scoring='f1_macro', n_jobs=-1)
            return cv_scores.mean()
    else:
        # Use early stopping with validation set
        if weighting_strategy != 'noweighting':
            sample_weights = np.array([class_weights[y] for y in y_train_part])
            model.fit(
                X_train_part.values, y_train_part.values, 
                sample_weight=sample_weights,
                eval_set=[(X_val.values, y_val.values)],
                verbose=False
            )
        else:
            model.fit(
                X_train_part.values, y_train_part.values,
                eval_set=[(X_val.values, y_val.values)],
                verbose=False
            )
        preds = model.predict(X_val.values)
        return f1_score(y_val, preds, average='macro')

# --------------------------
# CREATE STUDIES WITH PRUNING
# --------------------------
def create_study_with_defaults(model_name):
    """Create study with good defaults and pruning"""
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    
    # Enqueue good default parameters
    if model_name == "DecisionTree":
        study.enqueue_trial({
            'use_unlimited_depth': True,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini',
            'use_advanced_params': False
        })
    elif model_name == "RandomForest":
        study.enqueue_trial({
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'min_impurity_decrease': 0.0,
            'bootstrap': True
        })
    elif model_name == "XGBoost":
        study.enqueue_trial({
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'n_estimators': 800 if use_cv else 1000
        })
    
    return study

# --------------------------
# RUN OPTUNA STUDIES
# --------------------------
all_results = {}
models_objectives = [
    ("DecisionTree", objective_decision_tree),
    ("RandomForest", objective_random_forest),
    ("XGBoost", objective_xgboost)
]

for model_name, objective_fn in models_objectives:
    print(f"\n===== Optimizing {model_name} =====")
    print(f"Baseline to beat: {baseline_scores[model_name]:.4f}")
    
    study = create_study_with_defaults(model_name)
    study.optimize(objective_fn, n_trials=n_trials, timeout=1800)  # 30 minute timeout per model
    
    best_params = study.best_trial.params
    best_score = study.best_value
    print(f"Best validation score: {best_score:.4f}")
    print(f"Improvement over baseline: {best_score - baseline_scores[model_name]:.4f}")
    print(f"Best params for {model_name}: {best_params}")

    # Train final model on full training set with best parameters
    if model_name == "DecisionTree":
        # Handle unlimited depth
        if best_params.get('use_unlimited_depth', False):
            best_params['max_depth'] = None
        best_params.pop('use_unlimited_depth', None)
        best_params.pop('use_advanced_params', None)
        
        class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)} if weighting_strategy != 'noweighting' else None
        model = DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42, **best_params)
        model.fit(X_train_part, y_train_part)
        preds = model.predict(X_test)
        
    elif model_name == "RandomForest":
        class_weight_dict = {cls: w for cls, w in zip(unique_classes, class_weights)} if weighting_strategy != 'noweighting' else None
        model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42, n_jobs=-1, **best_params)
        model.fit(X_train_part, y_train_part)
        preds = model.predict(X_test)
        
    else:  # XGBoost
        model = XGBClassifier(**best_params)
        if weighting_strategy != 'noweighting':
            sample_weights = np.array([class_weights[y] for y in y_train_part])
            model.fit(X_train_part.values, y_train_part.values, sample_weight=sample_weights)
        else:
            model.fit(X_train_part.values, y_train_part.values)
        preds = model.predict(X_test.values)

    # Compute test metrics
    test_f1_macro = f1_score(y_test, preds, average='macro')
    test_f1_micro = f1_score(y_test, preds, average='micro')
    test_f1_weighted = f1_score(y_test, preds, average='weighted')
    test_precision_macro = precision_score(y_test, preds, average='macro')
    test_recall_macro = recall_score(y_test, preds, average='macro')
    conf_matrix = confusion_matrix(y_test, preds)
    class_report_dict = classification_report(y_test, preds, output_dict=True)

    # Save results
    all_results[model_name] = {
        "filter_size": FILTER_SIZE,
        "best_params": best_params,
        "best_validation_score": best_score,
        "baseline_validation_score": baseline_scores[model_name],
        "improvement_over_baseline": best_score - baseline_scores[model_name],
        "n_trials": n_trials,
        "use_cv": use_cv,
        "test_f1_macro": test_f1_macro,
        "test_f1_micro": test_f1_micro,
        "test_f1_weighted": test_f1_weighted,
        "test_precision_macro": test_precision_macro,
        "test_recall_macro": test_recall_macro,
        "cv_class_weights": imbalance_metrics["cv_class_weights"],
        "imbalance_ratio": imbalance_metrics["imbalance_ratio"],
        "entropy": imbalance_metrics["entropy"],
        "total_training_samples": len(X_train_part),
        "classification_reports": class_report_dict,
        "confusion_matrix": conf_matrix.tolist()
    }

# Save results
json_filename = f"improved_optuna_all_models_metrics_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json"
if use_cv:
    json_filename = json_filename.replace('.json', '_cv.json')

json_path = os.path.join(RESULT_PATH, json_filename)
with open(json_path, "w") as f:
    json.dump(convert_to_serializable(all_results), f, indent=4)

print(f"\nAll metrics saved to {json_path}")

# Print summary
print(f"\n{'='*50}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*50}")
for model_name in all_results:
    result = all_results[model_name]
    print(f"\n{model_name}:")
    print(f"  Baseline F1:     {result['baseline_validation_score']:.4f}")
    print(f"  Optimized F1:    {result['best_validation_score']:.4f}")
    print(f"  Improvement:     {result['improvement_over_baseline']:.4f}")
    print(f"  Test F1:         {result['test_f1_macro']:.4f}")

overall_end_time = time.time()
print(f"\nTotal runtime: {(overall_end_time - overall_start_time) / 60:.2f} minutes")