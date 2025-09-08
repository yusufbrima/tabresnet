import os 
from pathlib import Path 
import pandas as pd 
from config import EICU_PATH,OUTPUT_PATH
from datetime import datetime, timedelta


# Load main eICU tables
print("Loading eICU tables...")

# Core patient information
patient = pd.read_csv(f'{EICU_PATH}/patient.csv.gz')
print(f"Patient table: {patient.shape}")


# Admission information with outcomes
apache = pd.read_csv(f'{EICU_PATH}/apacheApsVar.csv.gz')  # APACHE scores and components
print(f"APACHE table: {apache.shape}")

# Vital signs (first 24 hours only to avoid leakage)
vitals = pd.read_csv(f'{EICU_PATH}/vitalPeriodic.csv.gz')
print(f"Vitals table: {vitals.shape}")

# Lab values (first 24 hours only to avoid leakage)
labs = pd.read_csv(f'{EICU_PATH}/lab.csv.gz')
print(f"Labs table: {labs.shape}")

# Diagnosis information
diagnosis = pd.read_csv(f'{EICU_PATH}/diagnosis.csv.gz')
print(f"Diagnosis table: {diagnosis.shape}")

print("Data loading completed.\n")



# =============================================================================
# DATA PREPROCESSING AND TEMPORAL FILTERING TO AVOID LEAKAGE
# =============================================================================

print("Preprocessing data to avoid temporal leakage...")

# Convert time columns to numeric for filtering
vitals['observationoffset'] = pd.to_numeric(vitals['observationoffset'], errors='coerce')
labs['labresultoffset'] = pd.to_numeric(labs['labresultoffset'], errors='coerce')

# CRITICAL: Only use data from first 24 hours to avoid leakage
# This ensures we don't use future information to predict outcomes

# Filter vitals to first 24 hours (1440 minutes)
vitals_24h = vitals[vitals['observationoffset'] <= 1440].copy()
print(f"Vitals filtered to 24h: {vitals_24h.shape}")

# Filter labs to first 24 hours
labs_24h = labs[labs['labresultoffset'] <= 1440].copy()
print(f"Labs filtered to 24h: {labs_24h.shape}")

# Filter diagnosis to admission diagnoses only (offset <= 0)
diagnosis['diagnosisoffset'] = pd.to_numeric(diagnosis['diagnosisoffset'], errors='coerce')
admission_diagnosis = diagnosis[diagnosis['diagnosisoffset'] <= 0].copy()
print(f"Admission diagnosis only: {admission_diagnosis.shape}")

# =============================================================================
# AGGREGATE FEATURES FROM TIME-SERIES DATA
# =============================================================================

print("Aggregating time-series features...")

# Aggregate vital signs for each patient (first 24h)
vital_features = vitals_24h.groupby('patientunitstayid').agg({
    'temperature': ['mean', 'std', 'min', 'max', 'count'],
    'sao2': ['mean', 'std', 'min', 'max', 'count'],
    'heartrate': ['mean', 'std', 'min', 'max', 'count'],
    'respiration': ['mean', 'std', 'min', 'max', 'count'],
    'systemicsystolic': ['mean', 'std', 'min', 'max', 'count'],
    'systemicdiastolic': ['mean', 'std', 'min', 'max', 'count'],
    'systemicmean': ['mean', 'std', 'min', 'max', 'count']
}).round(3)

# Flatten column names
vital_features.columns = ['_'.join(col).strip() for col in vital_features.columns]
vital_features.reset_index(inplace=True)
print(f"Vital features aggregated: {vital_features.shape}")

# Aggregate lab values for each patient (first 24h)
# Get most common lab tests
common_labs = labs_24h['labname'].value_counts().head(20).index.tolist()
labs_pivot = labs_24h[labs_24h['labname'].isin(common_labs)].pivot_table(
    index='patientunitstayid',
    columns='labname', 
    values='labresult',
    aggfunc=['mean', 'std', 'min', 'max', 'count']
)

# Flatten column names
labs_pivot.columns = ['_'.join([str(agg), str(lab)]).strip() for agg, lab in labs_pivot.columns]
labs_pivot.reset_index(inplace=True)
print(f"Lab features aggregated: {labs_pivot.shape}")

# Create diagnosis features (admission only)
diagnosis_features = admission_diagnosis.groupby('patientunitstayid')['diagnosisstring'].apply(
    lambda x: '; '.join(x.astype(str))
).reset_index()
diagnosis_features.rename(columns={'diagnosisstring': 'admission_diagnoses'}, inplace=True)

# Count number of admission diagnoses
diagnosis_count = admission_diagnosis.groupby('patientunitstayid').size().reset_index()
diagnosis_count.rename(columns={0: 'num_admission_diagnoses'}, inplace=True)

print(f"Diagnosis features created: {diagnosis_features.shape}")

# =============================================================================
# CREATE TARGET VARIABLES (OUTCOMES) - MULTI-CLASS CLASSIFICATION
# =============================================================================

print("Creating multi-class target variables...")

# Merge patient and apache data to get outcomes
patient_outcomes = patient.merge(apache, on='patientunitstayid', how='left')

# Convert discharge offset to hours for easier interpretation
patient_outcomes['unitdischargeoffset'] = pd.to_numeric(patient_outcomes['unitdischargeoffset'], errors='coerce')
patient_outcomes['los_hours'] = patient_outcomes['unitdischargeoffset']

# 1. MORTALITY RISK CATEGORIES (4 classes)
# Based on combination of ICU and hospital outcomes
def create_mortality_category(row):
    if row['unitdischargestatus'] == 'Expired':
        return 'ICU_Death'  # Died in ICU
    elif row['hospitaldischargestatus'] == 'Expired':
        return 'Hospital_Death'  # Survived ICU but died in hospital
    elif row['los_hours'] > 2880:  # >48 hours
        return 'Critical_Survivor'  # Long stay survivor (likely had complications)
    else:
        return 'Routine_Discharge'  # Short stay, routine discharge

patient_outcomes['mortality_risk_category'] = patient_outcomes.apply(create_mortality_category, axis=1)

# 2. LENGTH OF STAY CATEGORIES (5 classes)
# Based on clinical decision points and resource utilization
def create_los_category(hours):
    if pd.isna(hours):
        return 'Unknown'
    elif hours <= 24:
        return 'Very_Short'  # <1 day - observation/brief monitoring
    elif hours <= 72:
        return 'Short'       # 1-3 days - routine ICU care
    elif hours <= 168:
        return 'Standard'    # 3-7 days - typical complicated case
    elif hours <= 720:
        return 'Extended'    # 1-4 weeks - prolonged recovery
    else:
        return 'Prolonged'   # >4 weeks - chronic critical illness

patient_outcomes['los_category'] = patient_outcomes['los_hours'].apply(create_los_category)

# 3. CLINICAL SEVERITY CATEGORIES (5 classes)
# Based on organ support requirements and interventions
def create_severity_category(row):
    score = 0
    
    # Respiratory support
    if row.get('intubated', 0) == 1 or row.get('vent', 0) == 1:
        score += 2
    
    # Renal support
    if row.get('dialysis', 0) == 1:
        score += 2
        
    # Cardiovascular - using mean BP as proxy for shock
    mean_bp = pd.to_numeric(row.get('meanbp', 100), errors='coerce')
    if mean_bp < 60:
        score += 2
    elif mean_bp < 70:
        score += 1
        
    # Neurological - Glasgow Coma Scale components
    try:
        gcs_total = (pd.to_numeric(row.get('eyes', 4), errors='coerce') + 
                    pd.to_numeric(row.get('verbal', 5), errors='coerce') + 
                    pd.to_numeric(row.get('motor', 6), errors='coerce'))
        if gcs_total <= 8:
            score += 2
        elif gcs_total <= 12:
            score += 1
    except:
        pass
    
    # Metabolic - lactate proxy using pH
    ph = pd.to_numeric(row.get('ph', 7.4), errors='coerce')
    if ph < 7.25:
        score += 1
    
    # Map score to categories
    if score == 0:
        return 'Minimal'      # No organ support
    elif score <= 2:
        return 'Mild'         # Single organ dysfunction
    elif score <= 4:
        return 'Moderate'     # Multi-organ dysfunction
    elif score <= 6:
        return 'Severe'       # Severe multi-organ failure
    else:
        return 'Critical'     # Life-threatening multi-organ failure

patient_outcomes['severity_category'] = patient_outcomes.apply(create_severity_category, axis=1)

# 4. DISCHARGE DISPOSITION (4 classes)
# Clinical meaningful discharge categories
def create_discharge_category(hospital_status, unit_status, unit_location):
    # Handle deaths
    if unit_status == 'Expired' or hospital_status == 'Expired' or unit_location == 'Death':
        return 'Death'
    
    # Home or hospice
    elif hospital_status in ['Home', 'Hospice'] or unit_location == 'Home':
        return 'Home / Hospice'
    
    # Rehabilitation / skilled nursing / long-term care
    elif hospital_status in ['Rehabilitation', 'Skilled Nursing Facility', 'Long Term Acute Care'] \
         or unit_location in ['Rehabilitation', 'Skilled Nursing Facility', 'Long Term Acute Care']:
        return 'Extended Care'
    
    # ICU categories
    elif unit_location in ['ICU', 'Other ICU', 'Other ICU (CABG)']:
        return 'ICU'
    
    # Step-down / intermediate care
    elif unit_location in ['Step-Down Unit (SDU)', 'Telemetry']:
        return 'Step-Down / Telemetry'
    
    # Floor / acute care
    elif unit_location in ['Floor', 'Acute Care/Floor']:
        return 'Floor / Acute Care'
    
    # Other hospital or external transfer
    elif unit_location in ['Other Hospital', 'Other External', 'Other Internal', 'Operating Room']:
        return 'Other Hospital / External'
    
    # Nursing home
    elif unit_location in ['Nursing Home']:
        return 'Nursing Home'
    
    # Catch-all
    else:
        return 'Other'

# Apply function
patient_outcomes['discharge_category'] = patient_outcomes.apply(
    lambda row: create_discharge_category(
        row.get('hospitaldischargestatus', ''), 
        row.get('unitdischargestatus', ''),
        row.get('unitdischargelocation', '')
    ), axis=1
)

# Remove underscores just in case
patient_outcomes['discharge_category'] = patient_outcomes['discharge_category'].str.replace('_', ' ')


# def create_discharge_category(hospital_status, unit_status):
#     if unit_status == 'Expired' or hospital_status == 'Expired':
#         return 'Death'
#     elif hospital_status in ['Home', 'Hospice']:
#         return 'Home_Hospice'
#     elif hospital_status in ['Rehabilitation', 'Skilled Nursing Facility', 'Long Term Acute Care']:
#         return 'Extended_Care'
#     else:
#         return 'Other_Hospital'  # Transfer to another acute care facility

# patient_outcomes['discharge_category'] = patient_outcomes.apply(
#     lambda row: create_discharge_category(
#         row.get('hospitaldischargestatus', ''), 
#         row.get('unitdischargestatus', '')
#     ), axis=1
# )

# 5. RESOURCE UTILIZATION CATEGORIES (4 classes)
# Based on combination of LOS and interventions
def create_resource_category(row):
    los = row.get('los_hours', 0)
    has_dialysis = row.get('dialysis', 0) == 1
    has_ventilation = row.get('vent', 0) == 1 or row.get('intubated', 0) == 1
    
    if los <= 48 and not has_dialysis and not has_ventilation:
        return 'Low_Resource'
    elif los <= 168 and (has_dialysis or has_ventilation):
        return 'High_Intensity'
    elif los > 168 and (has_dialysis or has_ventilation):
        return 'Very_High_Resource'
    else:
        return 'Standard_Resource'

patient_outcomes['resource_category'] = patient_outcomes.apply(create_resource_category, axis=1)

# =============================================================================
# CREATE DIAGNOSIS-BASED TARGET VARIABLES
# =============================================================================

print("Creating diagnosis-based target variables...")

# Merge admission diagnosis data with patient outcomes
patient_outcomes = patient_outcomes.merge(
    admission_diagnosis[['patientunitstayid', 'diagnosisstring']], 
    on='patientunitstayid', 
    how='left'
)

# 6. PRIMARY DIAGNOSIS CATEGORIES (8 major categories)
def categorize_primary_diagnosis(diagnosis_text):
    if pd.isna(diagnosis_text):
        return 'Unknown'
    
    diagnosis = str(diagnosis_text).lower()
    
    # Cardiovascular conditions
    if any(keyword in diagnosis for keyword in ['cardiac', 'heart', 'myocardial', 'coronary', 'arrhythmia', 'cardiogenic']):
        return 'Cardiovascular'
    
    # Respiratory conditions  
    elif any(keyword in diagnosis for keyword in ['respiratory', 'pneumonia', 'copd', 'asthma', 'pulmonary', 'lung']):
        return 'Respiratory'
    
    # Neurological conditions
    elif any(keyword in diagnosis for keyword in ['neuro', 'stroke', 'seizure', 'brain', 'intracranial', 'coma']):
        return 'Neurological'
    
    # Sepsis and infections
    elif any(keyword in diagnosis for keyword in ['sepsis', 'septic', 'infection', 'bacteremia', 'pneumonia']):
        return 'Sepsis_Infection'
    
    # Gastrointestinal conditions
    elif any(keyword in diagnosis for keyword in ['gastrointestinal', 'liver', 'hepatic', 'bowel', 'abdominal', 'gi bleeding']):
        return 'Gastrointestinal'
    
    # Trauma and surgery
    elif any(keyword in diagnosis for keyword in ['trauma', 'fracture', 'surgery', 'post-op', 'surgical']):
        return 'Trauma_Surgery'
    
    # Renal conditions
    elif any(keyword in diagnosis for keyword in ['renal', 'kidney', 'acute kidney injury', 'dialysis']):
        return 'Renal'
    
    # Other medical conditions
    else:
        return 'Other_Medical'

# Group diagnoses by patient and take the first (primary) diagnosis
primary_diagnosis = admission_diagnosis.groupby('patientunitstayid')['diagnosisstring'].first().reset_index()
primary_diagnosis['primary_diagnosis_category'] = primary_diagnosis['diagnosisstring'].apply(categorize_primary_diagnosis)

# Merge primary diagnosis category
patient_outcomes = patient_outcomes.merge(
    primary_diagnosis[['patientunitstayid', 'primary_diagnosis_category']], 
    on='patientunitstayid', 
    how='left'
)

# 7. SEPSIS SEVERITY CATEGORIES (4 classes)
def categorize_sepsis_severity(row):
    diagnosis = str(row.get('diagnosisstring', '')).lower()
    
    # Check for sepsis indicators
    has_sepsis = any(keyword in diagnosis for keyword in ['sepsis', 'septic'])
    
    if not has_sepsis:
        return 'No_Sepsis'
    
    # Determine severity based on organ support and vital signs
    has_shock = False
    
    # Cardiovascular indicators
    mean_bp = pd.to_numeric(row.get('meanbp', 100), errors='coerce')
    if mean_bp < 65:
        has_shock = True
    
    # Renal indicators
    has_dialysis = row.get('dialysis', 0) == 1
    
    # Respiratory indicators
    has_ventilation = row.get('vent', 0) == 1 or row.get('intubated', 0) == 1
    
    # Categorize sepsis severity
    if 'severe sepsis' in diagnosis or 'septic shock' in diagnosis or has_shock:
        return 'Septic_Shock'
    elif has_dialysis or has_ventilation or 'severe' in diagnosis:
        return 'Severe_Sepsis'
    else:
        return 'Sepsis'

patient_outcomes['sepsis_severity'] = patient_outcomes.apply(categorize_sepsis_severity, axis=1)

# 8. CARDIAC CONDITIONS SUBCATEGORIES (5 classes)
def categorize_cardiac_conditions(diagnosis_text):
    if pd.isna(diagnosis_text):
        return 'Non_Cardiac'
    
    diagnosis = str(diagnosis_text).lower()
    
    if not any(keyword in diagnosis for keyword in ['cardiac', 'heart', 'myocardial', 'coronary']):
        return 'Non_Cardiac'
    
    # Acute coronary syndrome
    if any(keyword in diagnosis for keyword in ['myocardial infarction', 'mi', 'acute coronary', 'stemi', 'nstemi']):
        return 'Acute_Coronary'
    
    # Heart failure
    elif any(keyword in diagnosis for keyword in ['heart failure', 'cardiogenic shock', 'pulmonary edema']):
        return 'Heart_Failure'
    
    # Arrhythmias
    elif any(keyword in diagnosis for keyword in ['arrhythmia', 'atrial fibrillation', 'ventricular tachycardia', 'bradycardia']):
        return 'Arrhythmia'
    
    # Cardiac surgery
    elif any(keyword in diagnosis for keyword in ['cardiac surgery', 'cabg', 'valve replacement', 'post-cardiac']):
        return 'Cardiac_Surgery'
    
    else:
        return 'Other_Cardiac'

# Apply cardiac categorization to primary diagnosis
patient_outcomes['cardiac_condition_category'] = patient_outcomes['diagnosisstring'].apply(categorize_cardiac_conditions)

print("Multi-class target variables created.")

# Print distribution of each target variable
target_variables = ['mortality_risk_category', 'los_category', 'severity_category', 
                   'discharge_category', 'resource_category', 'primary_diagnosis_category',
                   'sepsis_severity', 'cardiac_condition_category']

for target in target_variables:
    print(f"\n{target} distribution:")
    print(patient_outcomes[target].value_counts().sort_index())
    print(f"Number of classes: {patient_outcomes[target].nunique()}")

# =============================================================================
# MERGE ALL FEATURES (NO ENCODING)
# =============================================================================

print("Merging all features...")

# Start with patient outcomes as base - keep original categorical values
final_dataset = patient_outcomes[['patientunitstayid', 'age', 'gender', 'ethnicity', 
                                'admissionheight', 'admissionweight', 
                                'mortality_risk_category', 'los_category', 'severity_category',
                                'discharge_category', 'resource_category', 'primary_diagnosis_category',
                                'sepsis_severity', 'cardiac_condition_category']].copy()

# Add APACHE features (these are calculated at admission, so safe to use)
apache_features = apache[['patientunitstayid', 'intubated', 'vent', 'dialysis', 'eyes', 'motor', 'verbal', 
                         'meds', 'urine', 'wbc', 'temperature', 'respiratoryrate', 'sodium', 
                         'heartrate', 'meanbp', 'ph', 'hematocrit', 'creatinine', 'albumin', 
                         'pao2', 'pco2', 'bun', 'glucose', 'bilirubin', 'fio2']].copy()
final_dataset = final_dataset.merge(apache_features, on='patientunitstayid', how='left')

# Merge vital signs features
final_dataset = final_dataset.merge(vital_features, on='patientunitstayid', how='left')

# Merge lab features
final_dataset = final_dataset.merge(labs_pivot, on='patientunitstayid', how='left')

# Merge diagnosis features
final_dataset = final_dataset.merge(diagnosis_features, on='patientunitstayid', how='left')
final_dataset = final_dataset.merge(diagnosis_count, on='patientunitstayid', how='left')

print(f"Final merged dataset shape: {final_dataset.shape}")

# =============================================================================
# DATA QUALITY AND LEAKAGE PREVENTION CHECKS
# =============================================================================

print("Performing data quality checks...")

# Check for data leakage indicators
leakage_columns = ['hospitaldischargestatus', 'unitdischargestatus', 'unitdischargeoffset']
present_leakage_cols = [col for col in leakage_columns if col in final_dataset.columns]
if present_leakage_cols:
    print(f"WARNING: Potential leakage columns found: {present_leakage_cols}")
    final_dataset = final_dataset.drop(columns=present_leakage_cols)

# Remove rows where target variables are missing
print(f"Rows before removing missing targets: {len(final_dataset)}")
final_dataset = final_dataset.dropna(subset=['mortality_risk_category', 'los_category'])
print(f"Rows after removing missing targets: {len(final_dataset)}")

# Show missing values summary
print("Missing values per column:")
missing_counts = final_dataset.isnull().sum()
missing_pct = (missing_counts / len(final_dataset) * 100).round(2)
missing_summary = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage': missing_pct
})
print(missing_summary[missing_summary['Missing_Count'] > 0])

# =============================================================================
# PREPARE FINAL DATASET (RAW VALUES)
# =============================================================================

# Select feature columns (exclude target variables and IDs)
feature_columns = [col for col in final_dataset.columns 
                  if col not in ['patientunitstayid', 'mortality_risk_category', 'los_category', 
                                'severity_category', 'discharge_category', 'resource_category',
                                'primary_diagnosis_category', 'sepsis_severity', 'cardiac_condition_category']]

target_columns = ['mortality_risk_category', 'los_category', 'severity_category', 
                 'discharge_category', 'resource_category', 'primary_diagnosis_category',
                 'sepsis_severity', 'cardiac_condition_category']

print(f"Number of feature columns: {len(feature_columns)}")
print(f"Number of target columns: {len(target_columns)}")

# Final dataset summary
print(f"\nFinal dataset summary:")
print(f"Total samples: {len(final_dataset)}")
print(f"Total features: {len(feature_columns)}")

# Show distribution of each target variable
for target in target_columns:
    print(f"\n{target} distribution:")
    target_dist = final_dataset[target].value_counts()
    for class_name, count in target_dist.items():
        pct = (count / len(final_dataset) * 100)
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    print(f"  Total classes: {final_dataset[target].nunique()}")

# Show data types
print(f"\nData types summary:")
print(final_dataset[feature_columns].dtypes.value_counts())

# =============================================================================
# SAVE PROCESSED DATASET
# =============================================================================

print("Saving processed dataset...")

# Save feature matrix and targets separately
features_df = final_dataset[['patientunitstayid'] + feature_columns]
targets_df = final_dataset[['patientunitstayid'] + target_columns]

# Save to CSV files
features_df.to_csv(f'{OUTPUT_PATH}/eicu_features.csv', index=False)
targets_df.to_csv(f'{OUTPUT_PATH}/eicu_targets.csv', index=False)
final_dataset.to_csv(f'{OUTPUT_PATH}/eicu_merged_dataset.csv', index=False)

print("Dataset saved successfully!")
print(f"- Features saved to: eicu_features.csv")
print(f"- Targets saved to: eicu_targets.csv") 
print(f"- Complete dataset saved to: eicu_merged_dataset.csv")

print("\nFeatures ready for your encoding pipeline!")
print("Categorical features preserved as original values:")
print("- gender (Male/Female/Other)")
print("- ethnicity (original categories)")
print("- admission_diagnoses (text strings)")

print("\nScript completed successfully!")