"""
Optuna hyperparameter search for TabNet (MIMIC only).
n_d and n_a are NOT searched and NOT passed — TabNet uses its internal defaults.
Usage:
    python run_optuna_tabnet_mimic.py --target_col icd_code_broad --filter_size 500 --n_trials 20 --n_epochs 200
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier

from data.dataset import TabularDataset
from utils import preprocess_mimic_data_advanced, get_class_weights, quantify_dataset_imbalance, convert_to_serializable
from config import OUTPUT_PATH, TABULAR_EPOCHS, VAL_SIZE, TEST_SIZE, TARGET_COL, PATIENCE, RESULT_PATH, CUTOFF, FILTER_SIZE, EXPERIMENT_ID

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

    # ensure numeric numpy arrays for TabNet
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    y_train = pd.to_numeric(y_train, errors="coerce").astype(np.int64)
    y_val   = pd.to_numeric(y_val, errors="coerce").astype(np.int64)
    y_test  = pd.to_numeric(y_test, errors="coerce").astype(np.int64)

    return X_train, X_val, X_test, y_train, y_val, y_test

def objective_factory(X_train_np, X_val_np, y_train_np, y_val_np, sample_weights, weighting_strategy, device_name, epochs, patience):
    """
    Returns objective(trial) closure capturing the data.
    Note: n_d and n_a are NOT passed — TabNet will use its internal defaults.
    """
    def objective(trial):
        # Only tune these three parameters
                # Learning rate - extended range
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        # Weight decay - extended range
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

        # Batch size - more options including larger sizes
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024, 2048])

        # instantiate TabNet WITHOUT n_d and n_a — let model defaults apply
        # Use default values for architectural parameters
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=lr, weight_decay=weight_decay),
            seed=RANDOM_SEED,
            device_name=device_name  # Pass the actual device name string
        )

        fit_kwargs = dict(
            X_train=X_train_np.to_numpy(),
            y_train=y_train_np.to_numpy(),
            eval_set=[(X_val_np.to_numpy(), y_val_np.to_numpy())],
            eval_name=['val'],
            eval_metric=['accuracy'],
            max_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            num_workers=2,
            drop_last=False
        )

        if weighting_strategy != 'noweighting' and sample_weights is not None:
            fit_kwargs['weights'] = sample_weights

        try:
            clf.fit(**fit_kwargs)
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return 0.0

        val_preds = clf.predict(X_val_np.to_numpy())
        try:
            val_f1 = f1_score(y_val_np.to_numpy(), val_preds, average='macro')
        except Exception:
            val_f1 = 0.0

        trial.report(val_f1, 0)
        return float(val_f1)
    return objective

def retrain_and_evaluate(best_params, X_train, X_val, X_test, y_train, y_val, y_test, sample_weights, weighting_strategy, epochs, patience, experiment_id, device_name):
    # combine train + val
    X_combined = pd.concat([X_train, X_val], ignore_index=True).to_numpy(dtype=np.float32)
    y_combined = np.concatenate([y_train.to_numpy(), y_val.to_numpy()])

    # instantiate TabNet WITHOUT n_d and n_a — model defaults used
    # Use default values for architectural parameters
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=best_params['lr'], weight_decay=best_params['weight_decay']),
        seed=RANDOM_SEED,
        device_name=device_name  # Pass the actual device name string
    )

    # vb = 128  # Use default virtual batch size
    # vb = min(vb, best_params.get('batch_size', 256))
    fit_kwargs = dict(
        X_train=X_combined,
        y_train=y_combined,
        eval_set=[(X_test.to_numpy(), y_test.to_numpy())],
        eval_name=['test'],
        eval_metric=['accuracy'],
        max_epochs=epochs,
        patience=patience,
        batch_size=best_params.get('batch_size', 256),
        num_workers=0,
        drop_last=False
    )
    if weighting_strategy != 'noweighting' and sample_weights is not None:
        _, combined_class_weights = get_class_weights(pd.Series(y_combined), strategy=weighting_strategy, beta=0.9999)
        combined_sw = np.array([combined_class_weights[int(y)] for y in y_combined])
        fit_kwargs['weights'] = combined_sw

    clf.fit(**fit_kwargs)

    final_model_dir = os.path.join(RESULT_PATH, f"tabnet_final_model_{experiment_id}")
    try:
        clf.save_model(final_model_dir)
    except Exception:
        import pickle
        final_model_path = os.path.join(RESULT_PATH, f"tabnet_model_{experiment_id}.pkl")
        with open(final_model_path, "wb") as fh:
            pickle.dump(clf, fh)
        final_model_dir = final_model_path

    test_preds = clf.predict(X_test.to_numpy())
    test_f1_macro = f1_score(y_test.to_numpy(), test_preds, average='macro')
    test_f1_micro = f1_score(y_test.to_numpy(), test_preds, average='micro')
    test_f1_weighted = f1_score(y_test.to_numpy(), test_preds, average='weighted')
    test_precision_macro = precision_score(y_test.to_numpy(), test_preds, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_test.to_numpy(), test_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_test.to_numpy(), test_preds)
    class_report_dict = classification_report(y_test.to_numpy(), test_preds, output_dict=True, zero_division=0)

    results = {
        "final_model_dir_or_path": final_model_dir,
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
    parser.add_argument('--weighting_strategy', type=str, default="inverse", choices=["inverse", "median", "effective", "noweighting"])
    parser.add_argument('--n_trials', type=int, default=40)
    parser.add_argument('--n_epochs', type=int, default=TABULAR_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    args = parser.parse_args()

    target_col = args.target_col
    filter_size = args.filter_size
    n_trials = args.n_trials
    epochs = args.n_epochs
    patience = args.patience
    weighting_strategy = args.weighting_strategy

    print(f"Running TabNet Optuna search on MIMIC | target: {target_col} | filter_size: {filter_size}")
    start_time = time.time()

    X_train, X_val, X_test, y_train, y_val, y_test = load_mimic(target_col, filter_size)

    class_weights, unique_classes = get_class_weights(y_train, strategy=weighting_strategy, beta=0.9999)
    class_counts = [np.sum(y_train == cls) for cls in unique_classes]
    imbalance_metrics = quantify_dataset_imbalance(class_counts=class_counts, class_weights=class_weights)
    print("Class weights:", class_weights)
    print("Imbalance metrics:", imbalance_metrics)

    sample_weights = np.array([class_weights[int(y)] for y in y_train.to_numpy()]) if weighting_strategy != 'noweighting' else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, " device_name:", device_name)

    # FIX: Pass device_name instead of device to objective_factory
    objective = objective_factory(X_train, X_val, y_train, y_val, sample_weights,
                              weighting_strategy, device_name, epochs, patience)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Optuna finished. Best trial params:")
    best_params = dict(study.best_trial.params)
    # ensure numeric types for JSON serialization
    best_params['lr'] = float(best_params.get("lr", 1e-4))
    best_params['weight_decay'] = float(best_params.get("weight_decay", 1e-6))
    best_params['batch_size'] = int(best_params.get("batch_size", 256))

    print(json.dumps(best_params, indent=2))

    # FIX: Pass device_name to retrain_and_evaluate
    final_results = retrain_and_evaluate(best_params, X_train, X_val, X_test, y_train, y_val, y_test,
                                    sample_weights, weighting_strategy, epochs, patience,
                                    EXPERIMENT_ID, device_name)

    all_results = {
        "experiment_id": EXPERIMENT_ID,
        "model": "TabNet",
        "dataset": "mimic",
        "target_col": target_col,
        "filter_size": filter_size,
        "weighting_strategy": weighting_strategy,
        "imbalance_metrics": imbalance_metrics,
        "optuna_best_params": best_params,
        "optuna_best_value": float(study.best_value),
        "final_results": final_results
    }

    json_filename = f"optuna_tabnet_mimic_results_{target_col}_{weighting_strategy}_{EXPERIMENT_ID}.json"
    json_path = os.path.join(RESULT_PATH, json_filename)
    with open(json_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    elapsed = (time.time() - start_time) / 60.0
    print(f"Saved results to: {json_path}")
    print(f"Total runtime: {elapsed:.2f} minutes")

if __name__ == "__main__":
    main()