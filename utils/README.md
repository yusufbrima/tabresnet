# An Investigation into the Robustness and Scalability of Machine Learning Models on ED and ICU Data Under Class Imbalance

## 1\. Project Overview

This repository contains the code for our research paper, "An Investigation into the Robustness and Scalability of Machine Learning Models on ED and ICU Data Under Class Imbalance." The project implements and compares a Tabular ResNet model (TabResNet) with various traditional machine learning models for clinical outcome prediction using data from large critical care databases.

-----

## 2\. Environment Setup

To ensure a consistent and reproducible environment, we recommend using Conda. This step **must be completed before any data processing or model training**.

### **2.1. Create and Activate Conda Environment**

```bash
conda create --name tabresnet python=3.9
conda activate tabresnet
```

### **2.2. Install Dependencies**

Navigate to the root directory of the project and install all required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

-----

## 3\. Dataset Setup and Preparation

The project supports data from the MIMIC-IV and eICU databases. Access to these datasets typically requires credentialing through PhysioNet.

### **3.1. Downloading the Data**

  * **MIMIC-IV ED**: Download the dataset from: [https://physionet.org/content/mimic-iv-ed/2.2/](https://physionet.org/content/mimic-iv-ed/2.2/)
  * **MIMIC-CXR**: Download the dataset from: [https://physionet.org/content/mimic-cxr/2.1.0/](https://physionet.org/content/mimic-cxr/2.1.0/)
  * **eICU Collaborative Research Database**: Download the dataset from: [https://physionet.org/content/eicu-crd/2.0/](https://physionet.org/content/eicu-crd/2.0/)

### **3.2. Processing the Data**

  * **MIMIC Data**: Use `processor_mimic.py` to process the MIMIC-IV ED data. If you are using the multimodal pathway with MIMIC-CXR, first execute `extract_files.py` to unzip the chest radiography files.
  * **eICU Data**: Use `processor_eicu.py` to process the eICU data.
  * **Create Final Dataset**: After processing, run `create_final.py` to generate the final dataset for model training.

-----

## 4\. Configuration

All project settings, including file paths, hyperparameters, and model configurations, are defined in `config.py`. This file must be reviewed and modified to match your system's data directory and specific experimental parameters.

-----

## 5\. Hyperparameter Search

Optuna is used to perform an efficient search for the best hyperparameters for each model. The search is conducted using the respective `run_optuna_...py` scripts.

  * The results of each search are stored as a JSON object in the `./results` directory. The naming convention is `optuna_tabnet_mimic_results_{target_col}_{weighting_strategy}_{EXPERIMENT_ID}.json`.

-----

## 6\. Model Training and Evaluation

After the hyperparameter search is complete, you can train and evaluate the models using the provided shell script.

### **6.1. Execute the Training Script**

First, make the script executable, then run it from the terminal.

```bash
chmod +x run_tabular_trainer.sh
./run_tabular_trainer.sh
```

### **6.2. Storing Results**

The final model performance metrics are saved as JSON objects in the `./results` directory, following the naming convention:
`all_models_metrics_traditional_models_{target_col}_{dataset_flag}_{weighting_strategy}_{experiment_id}.json`

-----

## 7\. Visualization and Analysis

The repository includes a Jupyter notebook, `plotting.ipynb`, to facilitate the visualization and analysis of experimental results.

  * The notebook loads the saved JSON objects from the `./results` directory.
  * It utilizes helper functions from `./utils/visualize.py` to generate publication-quality plots and charts for comparing model performance.

-----

## 8\. Copyright and License

**Author:** Yusuf Brima (Osnabrueck University)
**Copyright:** (c) 2025 Yusuf Brima. All Rights Reserved.

This project's code is licensed under the [Choose an appropriate open-source license, e.g., MIT, Apache 2.0]. You may use, modify, and distribute the code in accordance with the terms of this license.

### **Citation**

If you use this code in your research, please cite the following paper. The paper is currently under review at *Nature Digital Medicine*.

> Yusuf Brima. "An Investigation into the Robustness and Scalability of Machine Learning Models on ED and ICU Data Under Class Imbalance." Submitted to *Nature Digital Medicine*.
> *Submission guidelines:* [https://www.nature.com/collections/fjiebeacie/how-to-submit](https://www.nature.com/collections/fjiebeacie/how-to-submit)

Additionally, please cite the datasets as follows:

  * **MIMIC-CXR**:
    > Johnson, A., Pollard, T., Berkowitz, S., Greenbaum, N., Lungren, M., Deng, C., Mark, R., & Horng, S. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. *Scientific Data*, 6(1), 317.
  * **MIMIC-IV-ED**:
    > Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). *MIMIC-IV-ED* (version 2.2). PhysioNet.
  * **eICU Collaborative Research Database**:
    > Pollard, T. J., Johnson, A. E., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific Data*, 5, 180178.