import gzip
import shutil
from pathlib import Path
import pandas as pd
import os
from config import MIMIC_CXR_PATH, MIMIC_IV_ED_PATH, OUTPUT_PATH

## Unzipping the MIMC-IV-ED dataset


# Set your directory path
# input_dir = Path("physionet.org/files/mimic-iv-ed/2.2/ed")
input_dir = Path(MIMIC_IV_ED_PATH)

# Find all .gz files in the directory
gz_files = list(input_dir.glob("*.gz"))

# Iterate and extract each .gz file
for gz_file in gz_files:
    output_file = gz_file.with_suffix('')  # Remove .gz suffix
    print(f"Unzipping: {gz_file} → {output_file}")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


## Unzipping the MIMC-CXR dataset


# # Set your directory path
# input_dir = Path("Data/physionet.org/files/mimic-cxr/2.1.0/")
input_dir = Path(MIMIC_CXR_PATH)

# Find all .gz files in the directory
gz_files = list(input_dir.glob("*.gz"))

# Iterate and extract each .gz file
for gz_file in gz_files:
    output_file = gz_file.with_suffix('')  # Remove .gz suffix
    print(f"Unzipping: {gz_file} → {output_file}")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
