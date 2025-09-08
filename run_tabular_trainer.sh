#!/bin/bash

# target_cols=("icd_code_broad" "diagnosis" "disposition_grouped")
target_cols=('los_category' 'severity_category' 'discharge_category' 'resource_category')

# target_cols=("icd_code_broad" "diagnosis" "disposition_grouped")
weighting_strategies=("noweighting" "inverse" "effective" "median")
# weighting_strategies=("inverse")


for col in "${target_cols[@]}"
do
  for strategy in "${weighting_strategies[@]}"
  do
    echo "Running tab_trainer.py with target_col=${col} and weighting_strategy=${strategy}"
    python tab_trainer.py --target_col "$col" --weighting_strategy "$strategy"

    echo "Running tab_train_trad.py with target_col=${col} and weighting_strategy=${strategy}"
    python tab_train_trad.py --target_col "$col" --weighting_strategy "$strategy"

    echo "Running tab_train_tabnet.py with target_col=${col} and weighting_strategy=${strategy}"
    python tab_train_tabnet.py --target_col "$col" --weighting_strategy "$strategy"
  done
done



