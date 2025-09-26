#!/bin/bash

# --------------------------
# CONFIGURATION
# --------------------------
n_experiments=10
start_id=1

eicu_target_cols=("mortality_risk_category" "los_category" "severity_category" "discharge_category" "resource_category")
mimic_target_cols=("icd_code_broad" "diagnosis" "disposition_grouped")
weighting_strategies=("noweighting" "inverse" "effective" "median")
trainers=("tabresnet_trainer.py" "trad_ml_trainer.py" "tabnet_trainer.py")

# --------------------------
# BUILD JOB LIST
# --------------------------
job_file="jobs.txt"
> "$job_file"

for ((i=0; i<n_experiments; i++)); do
  exp_id=$((start_id + i))
  for dataset_flag in "eicu" "mimic"; do
    if [ "$dataset_flag" == "eicu" ]; then
      target_cols=("${eicu_target_cols[@]}")
    else
      target_cols=("${mimic_target_cols[@]}")
    fi

    for col in "${target_cols[@]}"; do
      for strategy in "${weighting_strategies[@]}"; do
        for trainer in "${trainers[@]}"; do
          echo "python $trainer --experiment_id $exp_id --random_seed $exp_id --dataset_flag $dataset_flag --target_col $col --weighting_strategy $strategy" >> "$job_file"
        done
      done
    done
  done
done

echo "Prepared job list with $(wc -l < $job_file) runs."

# --------------------------
# RUN JOBS IN PARALLEL
# --------------------------
# Change -P 8 to the number of jobs you want to run at once
cat jobs.txt | xargs -P 8 -I {} bash -c {}
