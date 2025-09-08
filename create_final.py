import os 
from pathlib import Path 
import pandas as pd 
from config import MIMIC_CXR_PATH, OUTPUT_PATH, MIMIC_CXR_JPG_PATH,FILTER_SIZE,VAL_SIZE,TEST_SIZE
from utils.helper import load_data, drop_columns, preprocess_mimic_data_advanced,create_diagnosis_column


#  (1) DICOM-based Preprocessing
target_col = 'diagnosis'  # Change this to the column you want to use as the target

# mimic_multimodal_image_centric_streamlined
# df_final = pd.read_csv(os.path.join(OUTPUT_PATH, 'mimic_multimodal_image_centric_advanced_streamlined.csv'))
df_final = pd.read_csv(os.path.join(OUTPUT_PATH, 'mimic_multimodal_image_centric_streamlined.csv'))

findings_classified = pd.read_csv(os.path.join(OUTPUT_PATH, 'mimic-cxr-2.0.0-chexpert-classified_streamlined.csv.gz'))

df_data = df_final.merge(findings_classified[['subject_id', 'study_id', 'diagnosis']], on=['subject_id', 'study_id'], how='left')


# drop values in diagnosis that are  Uncertain 
df_data = df_data[df_data['diagnosis'] != 'Uncertain']
# Reset index after filtering
df_data.reset_index(drop=True, inplace=True)

# Save the merged DataFrame to a CSV file
output_file = os.path.join(OUTPUT_PATH, 'mimic_cxr_multimodal_image_centric_diagnosis.csv')
df_data.to_csv(output_file, index=False)



processed_data = preprocess_mimic_data_advanced(
    output_path=OUTPUT_PATH,
    filename='mimic_cxr_multimodal_image_centric_diagnosis.csv',
    filter_size=None,
    target_col=target_col,
    impute_missing=True,
    test_size=0.2,
    val_size=0.2,
    random_state=42
)   



df_encoded = processed_data['final_df']

# print(df_encoded.columns)

# # Method 1: Loop through dataframe rows and collect all found files
found_rows = []  # List to store rows with found files

for index, row in df_encoded.iterrows():  # Fixed: use df_shuffled instead of df_final
    sample_path = Path(MIMIC_CXR_PATH, row['path'])
    
    if sample_path.is_file():
        # print(f"File exists: {sample_path}")
        # Add the row to our list of found rows
        found_rows.append(row)
        # found_rows.append(row.to_dict())  # Convert Series to dict to preserve all values


# Create new dataframe from found rows
df_found_files = pd.DataFrame(found_rows)

print(f"Original dataframe shape: {df_encoded.shape}")
print(f"Found files dataframe shape: {df_found_files.shape}")


out_file_name = Path(OUTPUT_PATH, "mimic_multimodal_image_centric_streamlined_found.csv")
df_found_files.to_csv(out_file_name, index=False)
print(f"\nDataset saved to: {out_file_name}")