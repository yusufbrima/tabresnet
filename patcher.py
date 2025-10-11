#!/usr/bin/env python3
import os, json, numpy as np
from utils.helper import preprocess_mimic_data_advanced, drop_columns, quantify_dataset_imbalance,\
    preprocess_eicu_data_advanced, get_class_weights, convert_to_serializable
from config import OUTPUT_PATH, VAL_SIZE, TEST_SIZE, CUTOFF, EICU_FILE

# --------------------------
# CONFIGURATION
# --------------------------
n_experiments = 10
start_id = 1

eicu_target_cols = ["mortality_risk_category", "los_category", "severity_category",
                    "discharge_category", "resource_category"]
mimic_target_cols = ["icd_code_broad", "diagnosis", "disposition_grouped"]
weighting_strategies = ["noweighting", "inverse", "effective", "median"]
trainers = ["tabresnet_trainer.py", "trad_ml_trainer.py", "tabnet_trainer.py"]

RESULT_PATH = "./results"
OUTPUT_PATH_JSON = "./outputs"
os.makedirs(OUTPUT_PATH_JSON, exist_ok=True)

# --------------------------
# HELPERS
# --------------------------
def compute_cvcf(class_counts):
    freqs = np.array(class_counts, dtype=float) / np.sum(class_counts)
    return float(np.std(freqs) / np.mean(freqs)) if np.mean(freqs) > 0 else None

def filename_for(trainer, target_col, dataset_flag, strategy, exp_id, aggregate=False):
    """Map trainer + flags → JSON filename"""
    if trainer == "tabresnet_trainer.py":
        return f"all_neural_models_metrics_{target_col}_{dataset_flag}_{strategy}_{exp_id}.json"
    elif trainer == "trad_ml_trainer.py":
        return f"all_models_metrics_traditional_models_{target_col}_{dataset_flag}_{strategy}_{exp_id}.json"
    elif trainer == "tabnet_trainer.py":
        if aggregate:
            return f"all_results_tabnet_results_filter_{target_col}_{dataset_flag}_{strategy}_{exp_id}.json"
        else:
            return f"all_single_tabnet_models_metrics_{target_col}_{dataset_flag}_{strategy}_{exp_id}.json"
    else:
        raise ValueError(f"Unknown trainer: {trainer}")

# --------------------------
# CACHE FOR CVCF VALUES
# --------------------------
cvcf_cache = {}

def precompute_cvcf(dataset_flag, target_col, exp_id, filter_sizes):
    """Precompute CVCF values for given filter_sizes and cache them."""
    key = (dataset_flag, target_col, exp_id)
    if key not in cvcf_cache:
        cvcf_cache[key] = {}

    for fs in filter_sizes:
        if fs in cvcf_cache[key]:
            continue  # already computed

        if dataset_flag == "mimic":
            processed = preprocess_mimic_data_advanced(
                output_path=OUTPUT_PATH,
                filename=CUTOFF if CUTOFF else 'mimic_multimodal_image_centric_streamlined_found.csv',
                filter_size=fs,
                target_col=target_col,
                impute_missing=True,
                staging=False,
                test_size=TEST_SIZE,
                val_size=VAL_SIZE,
                random_state=exp_id
            )
        else:
            processed = preprocess_eicu_data_advanced(
                output_path=OUTPUT_PATH,
                filename=EICU_FILE,
                target_col=target_col,
                impute_missing=True,
                filter_size=fs,
                test_size=TEST_SIZE,
                val_size=VAL_SIZE,
                random_state=exp_id
            )

        y_train = processed['y_train']
        class_counts = [np.sum(y_train == cls) for cls in np.unique(y_train)]
        cvcf_cache[key][fs] = compute_cvcf(class_counts)

def patch_file(fpath, dataset_flag, target_col, exp_id):
    """Load JSON, patch cv_class_weights using cached/precomputed values."""
    with open(fpath, "r") as f:
        results = json.load(f)

    for model_name, metrics in results.items():
        filter_sizes = metrics.get("filter_sizes", [])
        if not filter_sizes:
            continue

        # ensure values are precomputed
        precompute_cvcf(dataset_flag, target_col, exp_id, filter_sizes)

        # overwrite cv_class_weights
        key = (dataset_flag, target_col, exp_id)
        metrics["cv_class_weights"] = [cvcf_cache[key][fs] for fs in filter_sizes]

    return results

# --------------------------
# MAIN LOOP
# --------------------------
for exp_id in range(start_id, start_id + n_experiments):
    for dataset_flag in ["eicu", "mimic"]:
        target_cols = eicu_target_cols if dataset_flag == "eicu" else mimic_target_cols
        for target_col in target_cols:
            for strategy in weighting_strategies:
                for trainer in trainers:
                    # single model JSON
                    fname = filename_for(trainer, target_col, dataset_flag, strategy, exp_id, aggregate=False)
                    fpath = os.path.join(RESULT_PATH, fname)
                    if os.path.exists(fpath):
                        print(f"Patching {fname}")
                        results = patch_file(fpath, dataset_flag, target_col, exp_id)
                        outpath = os.path.join(OUTPUT_PATH_JSON, fname)
                        with open(outpath, "w") as f:
                            json.dump(results, f, indent=4)
                        print(f"✔ Saved patched {outpath}")

                    # aggregate tabnet results JSON
                    if trainer == "tabnet_trainer.py":
                        fname = filename_for(trainer, target_col, dataset_flag, strategy, exp_id, aggregate=True)
                        fpath = os.path.join(RESULT_PATH, fname)
                        if os.path.exists(fpath):
                            print(f"Patching {fname}")
                            results = patch_file(fpath, dataset_flag, target_col, exp_id)
                            outpath = os.path.join(OUTPUT_PATH_JSON, fname)
                            with open(outpath, "w") as f:
                                json.dump(results, f, indent=4)
                            print(f"✔ Saved patched {outpath}")

print("All done.")