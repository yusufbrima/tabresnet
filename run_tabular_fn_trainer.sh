#!/bin/bash

# --------------------------
# CONFIGURATION
# --------------------------
n_experiments=1  # total number of experiments
start_id=1        # starting experiment id

# Target columns for each dataset
# eicu_target_cols=("los_category" "severity_category" "discharge_category" "resource_category")
# eicu_target_cols=("mortality_risk_category" "los_category" "severity_category" "discharge_category" "resource_category")
eicu_target_cols=("mortality_risk_category")
mimic_target_cols=("icd_code_broad" "diagnosis" "disposition_grouped")

# Weighting strategies
weighting_strategies=("noweighting" "inverse" "effective" "median")

# --------------------------
# MAIN LOOP
# --------------------------
for ((i=0; i<n_experiments; i++)); do
  exp_id=$((start_id + i))
  echo "========== Running Experiment $exp_id =========="

  for dataset_flag in "eicu" "mimic"; do
    echo "---- Dataset: $dataset_flag ----"

    if [ "$dataset_flag" == "eicu" ]; then
      target_cols=("${eicu_target_cols[@]}")
    else
      target_cols=("${mimic_target_cols[@]}")
    fi

    for col in "${target_cols[@]}"; do
      for strategy in "${weighting_strategies[@]}"; do
        echo "Running tabicl_trainer.py with exp_id=${exp_id}, dataset=${dataset_flag}, target_col=${col}, weighting_strategy=${strategy}"
        python tabicl_trainer.py --experiment_id "$exp_id" --random_seed "$exp_id" --dataset_flag "$dataset_flag" --target_col "$col" --weighting_strategy "$strategy"
      done
    done
  done
done