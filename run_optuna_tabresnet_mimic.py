"""
Optuna hyperparameter search + final training/evaluation using ONLY the MIMIC dataset.
Usage example:
    python run_optuna_mimic.py --target_col icd_code_broad --filter_size 500 --n_trials 100
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

# repo imports (keep your implementations)
from models.TabularClassifier import AdvancedTabularClassifier
from models.trainer import tabevaluate, tabtrain
from data.dataset import TabularDataset
from utils.helper import preprocess_mimic_data_advanced, get_class_weights, quantify_dataset_imbalance, convert_to_serializable
from config import OUTPUT_PATH, TABULAR_EPOCHS, VAL_SIZE, TEST_SIZE, TARGET_COL, PATIENCE, RESULT_PATH, CUTOFF, FILTER_SIZE, EXPERIMENT_ID

# reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_mimic(target_col, filter_size):
    processed = preprocess_mimic_data_advanced(
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

    X_train = processed['X_train']
    X_val = processed['X_val']
    X_test = processed['X_test']
    y_train = processed['y_train']
    y_val = processed['y_val']
    y_test = processed['y_test']

    # drop irrelevant columns depending on target
    if target_col == 'diagnosis':
        cols_to_drop = ['icd_code_broad', 'disposition_grouped']
    elif target_col == 'disposition_grouped':
        cols_to_drop = ['icd_code_broad', 'diagnosis']
    elif target_col == 'icd_code_broad':
        cols_to_drop = ['diagnosis', 'disposition_grouped']
    else:
        cols_to_drop = []

    for df in [X_train, X_val, X_test]:
        if hasattr(df, "drop"):
            df.drop(columns=cols_to_drop + ['path'], inplace=True, errors='ignore')

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    def to_np(x):
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=np.float32)
        elif isinstance(x, np.ndarray):
            return x.astype(np.float32)
        else:
            return np.array(x, dtype=np.float32)

    X_train_np = to_np(X_train)
    X_val_np = to_np(X_val)
    X_test_np = to_np(X_test)

    train_loader = DataLoader(TabularDataset(X_train_np, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val_np, y_val), batch_size=batch_size)
    test_loader = DataLoader(TabularDataset(X_test_np, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader

def run_optuna(X_train, X_val, X_test, y_train, y_val, y_test, input_dim, output_dim, class_weights_tensor,
               device, n_trials, epochs, patience, weighting_strategy):
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", max(8, input_dim // 2), max(input_dim * 2, 16))
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        # dropout search removed — we fix dropout to 0.0 because you said not to search it
        dropout = 0.0
        # Learning rate - extended range
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

        # Weight decay - extended range
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

        # Batch size - more options including larger sizes
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024, 2048])

        # Use reduction - keep as is
        use_reduction = trial.suggest_categorical("use_reduction", [True, False])


        model = AdvancedTabularClassifier(input_dim, output_dim, hidden_dim=hidden_dim,
                                          num_blocks=num_blocks, use_reduction=use_reduction, dropout=dropout)
        model.to(device)

        train_loader, val_loader, _ = build_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size)

        if weighting_strategy == 'noweighting':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_f1 = -1.0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, _ = tabtrain(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = tabevaluate(model, val_loader, criterion, device)

            try:
                val_f1_macro = f1_score(val_labels, val_preds, average='macro')
            except Exception:
                val_f1_macro = float(val_acc)

            trial.report(val_f1_macro, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            scheduler.step(val_loss)

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                patience_counter = 0
                # save checkpoint for this trial (optional)
                torch.save(model.state_dict(), f"trial_{trial.number}_best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val_f1

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def final_train_and_eval(best_params, X_train, X_val, X_test, y_train, y_val, y_test, input_dim, output_dim,
                         class_weights_tensor, device, epochs, weighting_strategy, experiment_id):
    # combine train + val for final training
    X_train_combined = pd.concat([pd.DataFrame(X_train) if isinstance(X_train, (pd.DataFrame, pd.Series)) else pd.DataFrame(X_train),
                                  pd.DataFrame(X_val) if isinstance(X_val, (pd.DataFrame, pd.Series)) else pd.DataFrame(X_val)],
                                 ignore_index=True).to_numpy(dtype=np.float32)
    y_train_combined = np.concatenate([np.array(y_train), np.array(y_val)])

    batch_size = best_params.get("batch_size", 256)
    train_loader = DataLoader(TabularDataset(X_train_combined, y_train_combined), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TabularDataset(
        X_test.to_numpy(dtype=np.float32) if isinstance(X_test, pd.DataFrame) else X_test, y_test), batch_size=batch_size)

    # use dropout from best_params if present, otherwise default to 0.0
    dropout_value = best_params.get("dropout", 0.0)

    model = AdvancedTabularClassifier(input_dim, output_dim,
                                      hidden_dim=best_params.get("hidden_dim"),
                                      num_blocks=best_params.get("num_blocks"),
                                      use_reduction=best_params.get("use_reduction"),
                                      dropout=dropout_value)
    model.to(device)

    if weighting_strategy == 'noweighting':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=best_params.get("lr", 1e-4), weight_decay=best_params.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(epochs):
        train_loss, _ = tabtrain(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)

    final_model_path = f"best_mimic_model_{experiment_id}.pth"
    torch.save(model.state_dict(), final_model_path)

    test_loss, test_acc, test_preds, test_labels = tabevaluate(model, test_loader, criterion, device)

    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_micro = f1_score(test_labels, test_preds, average='micro')
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    test_precision_macro = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_recall_macro = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(test_labels, test_preds)
    class_report_dict = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)

    results = {
        "final_model_path": final_model_path,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_micro": float(test_f1_micro),
        "test_f1_weighted": float(test_f1_weighted),
        "test_precision_macro": float(test_precision_macro),
        "test_recall_macro": float(test_recall_macro),
        "classification_report": class_report_dict,
        "confusion_matrix": conf_matrix.tolist()
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_col', type=str, default=TARGET_COL)
    parser.add_argument('--filter_size', type=int, default=(FILTER_SIZE[0] if isinstance(FILTER_SIZE, list) and len(FILTER_SIZE)>0 else None))
    parser.add_argument('--weighting_strategy', type=str, default="inverse", choices=["inverse", "effective", "median", "noweighting"])
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--n_epochs', type=int, default=TABULAR_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    args = parser.parse_args()

    target_col = args.target_col
    filter_size = args.filter_size
    n_trials = args.n_trials
    epochs = args.n_epochs
    patience = args.patience
    weighting_strategy = args.weighting_strategy

    print(f"Running on MIMIC only | target: {target_col} | filter_size: {filter_size}")

    start_time = time.time()

    X_train, X_val, X_test, y_train, y_val, y_test = load_mimic(target_col, filter_size)

    class_weights, unique_classes = get_class_weights(y_train, strategy=weighting_strategy, beta=0.9999)
    class_counts = [np.sum(y_train == cls) for cls in unique_classes]
    imbalance_metrics = quantify_dataset_imbalance(class_counts=class_counts, class_weights=class_weights)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("Class weights:", class_weights)
    print("Imbalance metrics:", imbalance_metrics)

    input_dim = X_train.shape[1] if hasattr(X_train, "shape") else np.array(X_train).shape[1]
    output_dim = len(np.unique(y_train))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    study = run_optuna(X_train, X_val, X_test, y_train, y_val, y_test,
                       input_dim, output_dim, class_weights_tensor, device,
                       n_trials=n_trials, epochs=epochs, patience=patience, weighting_strategy=weighting_strategy)

    print("Optuna best params:", study.best_trial.params)

    final_results = final_train_and_eval(study.best_trial.params, X_train, X_val, X_test, y_train, y_val, y_test,
                                        input_dim, output_dim, class_weights_tensor, device,
                                        epochs=epochs, weighting_strategy=weighting_strategy, experiment_id=EXPERIMENT_ID)

    all_results = {
        "experiment_id": EXPERIMENT_ID,
        "dataset": "mimic",
        "target_col": target_col,
        "filter_size": filter_size,
        "weighting_strategy": weighting_strategy,
        "imbalance_metrics": imbalance_metrics,
        "optuna_best_params": study.best_trial.params,
        "optuna_best_value": float(study.best_value),
        "final_results": final_results
    }

    json_filename = f"optuna_mimic_results_{target_col}_{weighting_strategy}_{EXPERIMENT_ID}.json"
    json_path = os.path.join(RESULT_PATH, json_filename)
    with open(json_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    elapsed = (time.time() - start_time) / 60.0
    print(f"Saved results to: {json_path}")
    print(f"Total runtime: {elapsed:.2f} minutes")

if __name__ == "__main__":
    main()