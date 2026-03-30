MIMIC_IV_ED_PATH = '/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/physionet.org/files/mimic-iv-ed/2.2/ed/'

MIMIC_CXR_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/physionet.org/files/mimic-cxr/2.1.0"

MIMIC_CXR_JPG_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/physionet.org/files/mimic-cxr-jpg/2.1.0"

EICU_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/physionet.org/files/eicu-crd/2.0"

VISION_EPOCHS  = 20
MULTIMODAL_EPOCHS = 1 
TABULAR_EPOCHS = 300


TARGET_COL = 'icd_code_broad' # icd_code_broad, diagnosis, disposition_grouped

VAL_SIZE = 0.2
TEST_SIZE = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10

BATCH_SIZE = 32



MIMIC_TARGETS = ['disposition_grouped','icd_code_broad','diagnosis']

OUTPUT_PATH  = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Data"
RESULT_PATH = "./results" 
MODEL_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/MXAI/models"
FIG_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/MXAI/figures"
LOOK_UP_TABLE = {3: '786', 2: '780', 6: 'R07', 1: '486', 4: '789', 5: 'J18', 0: '428'}
CUTOFF = "mimic_multimodal_image_centric_streamlined_found_fixed.csv"
EICU_FILE = 'eicu_merged_dataset.csv'

EICU_TARGETS = ['mortality_risk_category', 'los_category', 'severity_category','discharge_category','resource_category']
MIMIC_TARGETS = ['disposition_grouped','icd_code_broad','diagnosis']


EICU_TARGET_LABELS = {
    "mortality_risk_category": "Mortality Risk Prediction",
    "los_category": "Length of Stay Prediction",
    "severity_category": "Severity Category Prediction",
    "discharge_category": "Discharge Category Prediction",
    "resource_category": "Resource Utilization Prediction"
}

MIMIC_TARGET_LABELS = {
    "disposition_grouped": "Disposition Prediction",
    "icd_code_broad": "ICD Code Prediction",
    "diagnosis": "Diagnosis Prediction"
}

DATASET_FLAG = 'eicu' #'mimic', 'eicu'

if DATASET_FLAG == 'eicu':
    # Filter Sizes for EICU
    FILTER_SIZE = [300, 1000, 1500, 3500, 4500, 6000, 8000, 12000]
else:
    # Filter Sizes for MIMIC-IV 
    FILTER_SIZE = [500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250, 3500, 3750, 4000]


EXPERIMENT_ID = 2
RANDOM_SEED = 42

