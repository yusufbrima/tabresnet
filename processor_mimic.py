import pandas as pd
import os
from utils.helper import lookup_icd_code
from config import MIMIC_CXR_PATH, MIMIC_IV_ED_PATH, OUTPUT_PATH

print("--- Loading MIMIC Data for Multimodal Diagnosis Task ---")

# --- 1. Load MIMIC-CXR Core Data ---
print("Loading MIMIC-CXR data...")
cxr_record_list_df = pd.read_csv(os.path.join(MIMIC_CXR_PATH, 'cxr-record-list.csv.gz'))
print(f"Loaded {cxr_record_list_df.shape[0]} images")

# Start with images as the base DataFrame
linked_df = cxr_record_list_df.copy()
image_counts = {'initial_images': linked_df.shape[0]}

# --- 2. Load MIMIC-IV-ED Data ---
print("\nLoading MIMIC-IV-ED data...")
edstays_df = pd.read_csv(os.path.join(MIMIC_IV_ED_PATH, 'edstays.csv'))
diagnosis_df = pd.read_csv(os.path.join(MIMIC_IV_ED_PATH, 'diagnosis.csv'))
triage_df = pd.read_csv(os.path.join(MIMIC_IV_ED_PATH, 'triage.csv'))
vitalsign_df = pd.read_csv(os.path.join(MIMIC_IV_ED_PATH, 'vitalsign.csv'))

print(f"Loaded ED stays: {edstays_df.shape[0]}")
print(f"Loaded diagnoses: {diagnosis_df.shape[0]}")
print(f"Loaded triage: {triage_df.shape[0]}")
print(f"Loaded vital signs: {vitalsign_df.shape[0]}")

# --- 3. Process Diagnoses and Dispositions ---
print("\nProcessing diagnoses and dispositions...")
diagnosis_df['icd_code_broad'] = diagnosis_df['icd_code'].astype(str).str[:3]
primary_diagnosis = diagnosis_df.groupby(['subject_id', 'stay_id']).first().reset_index()

disposition_mapping = {
    'ADMITTED': 'Admitted',
    'TRANSFER': 'Transferred',
    'EXPIRED': 'Deceased',
    'HOME': 'Discharged',
    'LEFT WITHOUT BEING SEEN': 'Discharged',
    'LEFT AGAINST MEDICAL ADVICE': 'Discharged',
    'ELOPED': 'Eloped',
    'OTHER': 'Other'
}
edstays_df['disposition_grouped'] = edstays_df['disposition'].map(disposition_mapping)

# --- 4. Aggregate Vital Signs ---
print("\nAggregating vital signs...")
vitalsign_agg = vitalsign_df.groupby('stay_id').agg({
    'temperature': 'mean',
    'heartrate': 'mean',
    'resprate': 'mean',
    'o2sat': 'mean',
    'sbp': 'mean',
    'dbp': 'mean'
}).reset_index()

vitalsign_agg.columns = ['stay_id', 'temperature_mean', 'heartrate_mean', 
                        'resprate_mean', 'o2sat_mean', 'sbp_mean', 'dbp_mean']

# --- 5. Combine All ED Data First ---
print("\nCombining all ED data tables...")
ed_data_combined = pd.merge(edstays_df, primary_diagnosis, on=['subject_id', 'stay_id'], how='left')
ed_data_combined = pd.merge(ed_data_combined, triage_df.drop(columns='chiefcomplaint', errors='ignore'), on=['subject_id', 'stay_id'], how='left')
ed_data_combined = pd.merge(ed_data_combined, vitalsign_agg, on='stay_id', how='left')
print(f"Combined ED data shape: {ed_data_combined.shape}")

# --- 6. Correct Linking: Link CXR Images to Combined ED Data ---
print("\nLinking images to combined ED data...")
# Use 'hadm_id' as the primary key for linking images to hospital encounters
# This is a more robust link than using 'subject_id' alone.
training_df = pd.merge(linked_df, ed_data_combined, on=['subject_id'], how='inner')
print(f"Final linked dataset: {training_df.shape[0]} images")

# --- 7. Clean and Prepare Final Dataset ---
print("\nPreparing final dataset...")
# Drop duplicates based on the image file
training_df = training_df.drop_duplicates(subset=['dicom_id'])

# Keep only rows that have a diagnosis from the ED stay
training_df = training_df[training_df['icd_code'].notna()].copy()

# Select final features
feature_columns = [
    # Identifiers
    'dicom_id', 'subject_id', 'study_id', 'stay_id', 'hadm_id', 'path',
    
    # Target variables
    'icd_code', 'icd_code_broad', 'icd_title', 'icd_version','disposition','disposition_grouped',
    
    # Demographics & ED info
    'gender', 'race', 'arrival_transport', 'anchor_age', 'anchor_year_group',
    
    # Triage vital signs
    'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity',
    
    # Aggregated vital signs
    'temperature_mean', 'heartrate_mean', 'resprate_mean', 
    'o2sat_mean', 'sbp_mean', 'dbp_mean'
]

# Keep only available columns
available_features = [col for col in feature_columns if col in training_df.columns]
training_df = training_df[available_features].copy()

print(f"Final training dataset: {training_df.shape[0]} images with {training_df.shape[1]} features")

# --- 8. Summary ---
final_count = training_df.shape[0]
retention_rate = (final_count / image_counts['initial_images']) * 100

print(f"\n=== SUMMARY ===")
print(f"Initial images: {image_counts['initial_images']:,}")
print(f"Final training images: {final_count:,}")
print(f"Retention rate: {retention_rate:.1f}%")
print(f"Sample: {training_df[['dicom_id', 'icd_code_broad', 'acuity']].head().to_string()}")

# --- 9. Save Dataset ---
output_file = os.path.join(OUTPUT_PATH, 'mimic_multimodal_image_centric_streamlined_fixed.csv')
training_df.to_csv(output_file, index=False)
print(f"\nDataset saved to: {output_file}")