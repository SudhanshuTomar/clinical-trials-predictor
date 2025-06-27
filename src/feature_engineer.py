import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# -----------------------------------
# Column Cleaning
# -----------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names by:
    - Stripping whitespace
    - Lowercasing all names
    - Replacing spaces with underscores
    - Removing special characters

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame with updated column names
    """
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(' ', '_')
          .str.replace(r'[^\w_]', '', regex=True)
    )
    return df

# -----------------------------------
# Status Features
# -----------------------------------
def map_status_group(status: str) -> str:
    """
    Maps raw study status to simplified status group.

    Parameters:
        status (str): Original study status string

    Returns:
        str: Grouped label - 'approved', 'failed', 'ongoing', or 'unknown'
    """
    s = str(status).upper()
    if s == 'COMPLETED': return 'approved'
    if s in ['TERMINATED','WITHDRAWN','SUSPENDED']: return 'failed'
    if 'RECRUITING' in s or 'ACTIVE' in s: return 'ongoing'
    return 'unknown'

def create_status_group(df: pd.DataFrame, col: str='study_status') -> pd.DataFrame:
    """
    Adds a new column 'status_group' to the DataFrame using mapped study status.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        col (str): Column containing original status values

    Returns:
        pd.DataFrame: Updated DataFrame with 'status_group' column
    """

    df = df.copy()
    df[col] = df[col].fillna('')
    df['status_group'] = df[col].apply(map_status_group)
    return df

# -----------------------------------
# Date-based Features
# -----------------------------------
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features based on trial timeline dates:
    - Durations between milestones
    - Time gaps (lags)
    - Start year/month

    Parameters:
        df (pd.DataFrame): DataFrame with relevant date columns

    Returns:
        pd.DataFrame: Updated DataFrame with new date features
    """

    df = df.copy()
    # ensure datetime
    dates = ['start_date','primary_completion_date','completion_date',
             'first_posted','results_first_posted','last_update_posted']
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    # durations
    if 'start_date' in df.columns and 'completion_date' in df.columns:
        df['duration_days'] = (df['completion_date'] - df['start_date']).dt.days.fillna(0)
    if 'start_date' in df.columns and 'primary_completion_date' in df.columns:
        df['time_to_primary_days'] = (df['primary_completion_date'] - df['start_date']).dt.days.fillna(0)
    # lags
    if 'first_posted' in df.columns and 'start_date' in df.columns:
        df['time_to_start'] = (df['start_date'] - df['first_posted']).dt.days.fillna(0)
    if 'results_first_posted' in df.columns and 'start_date' in df.columns:
        df['time_to_results'] = (df['results_first_posted'] - df['start_date']).dt.days.fillna(0)
    if 'last_update_posted' in df.columns and 'start_date' in df.columns:
        df['time_to_last_update'] = (df['last_update_posted'] - df['start_date']).dt.days.fillna(0)
    # extract year and month
    if 'start_date' in df.columns:
        df['start_year'] = df['start_date'].dt.year
        df['start_month'] = df['start_date'].dt.month
    return df

# -----------------------------------
# Sponsor Features
# -----------------------------------
def create_sponsor_approval_rate(df: pd.DataFrame, sponsor_rates: pd.Series = None) -> pd.DataFrame:
    """
    Calculates sponsor-specific approval rate and merges into the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'sponsor' and 'outcome' columns

    Returns:
        pd.DataFrame: Updated DataFrame with 'sponsor_approval_rate'
    """
    df = df.copy()
    if sponsor_rates is None:
        # FALLBACK: compute on-the-fly (just as before)
        df['outcome_numeric'] = df['outcome'].map({'Approved': 1, 'Failed': 0})
        rates = df.groupby('sponsor')['outcome_numeric'].mean().rename('sponsor_approval_rate')
    else:
        rates = sponsor_rates
    df['sponsor_approval_rate'] = df['sponsor'].map(rates).fillna(rates.mean())
    return df.drop(columns=['outcome_numeric'], errors='ignore')

# -----------------------------------
# Location Features
# -----------------------------------
def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a numeric feature counting number of locations in each trial.

    Parameters:
        df (pd.DataFrame): DataFrame with 'locations' column

    Returns:
        pd.DataFrame: Updated DataFrame with 'n_locations'
    """

    df = df.copy()
    if 'locations' in df.columns:
        # count pipe-delimited locations
        df['n_locations'] = df['locations'].astype(str).str.count(r'\|').add(1).fillna(1).astype(int)
    return df

# -----------------------------------
# Intervention Features
# -----------------------------------
def create_intervention_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from the 'interventions' column:
    - Total count
    - Type breakdown (drug, device, etc.)
    - Chemotherapy presence
    - Number of unique drug names

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Enriched DataFrame with intervention-related features
    """

    df = df.copy()
    if 'interventions' in df.columns:
        # count total interventions
        df['n_interventions'] = df['interventions'].astype(str).str.count(r'\|').add(1)
        # explode for type counts
        expl = (
            df[['nct_number','interventions']]
              .assign(intervention=lambda d: d['interventions'].astype(str).str.split('|'))
              .explode('intervention')
        )
        expl['type'] = expl['intervention'].str.extract(r'^([^:]+):', expand=False).str.strip().fillna('other')
        type_counts = (
            expl.groupby(['nct_number','type'])
                 .size()
                 .unstack(fill_value=0)
                 .add_prefix('n_type_')
                 .reset_index()
        )
        df = df.merge(type_counts, on='nct_number', how='left')
        # presence flags example
        df['has_chemotherapy'] = df['interventions'].str.contains(
            'chemotherapy|cisplatin|doxorubicin|paclitaxel',
            case=False, na=False).astype(int)
        # count unique drugs
        expl['drug_name'] = expl['intervention'].str.replace(r'^[^:]+:\s*','', regex=True).str.lower()
        unique_counts = (
            expl.groupby('nct_number')['drug_name']
                 .nunique()
                 .rename('n_unique_interventions')
                 .reset_index()
        )
        df = df.merge(unique_counts, on='nct_number', how='left')
    return df

# -----------------------------------
# Therapeutic Area Features
# -----------------------------------
therapeutic_area_keywords = {
    "Oncology": [
        "cancer", "carcinoma", "neoplasm", "tumor", "sarcoma", "lymphoma", "leukemia",
        "myeloma", "melanoma", "angiosarcoma", "blastoma", "glioma", "mesothelioma",
        "adenocarcinoma", "metastasis", "oncology", "HPV", "sarcomas", "myelodysplastic",
        "GIST", "neuroblastoma", "osteosarcoma", "retinoblastoma", "teratoma", "hepatocellular"
    ],

    "Cardiology": [
        "heart", "cardiac", "cardiovascular", "hypertension", "high blood pressure",
        "arrhythmia", "myocardial", "infarction", "heart failure", "angina", "atherosclerosis",
        "stroke", "valve", "pericarditis", "endocarditis", "tachycardia", "bradycardia",
        "ventricular", "atrial", "ischemia", "heart transplant"
    ],

    "Neurology": [
        "alzheimer", "parkinson", "multiple sclerosis", "ms", "epilepsy", "seizure",
        "stroke", "cerebral", "dementia", "migraine", "neuropathy", "ataxia", "encephalitis",
        "encephalopathy", "ALS", "amyotrophic", "myasthenia gravis", "huntington",
        "guillain", "chorea", "headache", "neurosurgery"
    ],

    "Infectious Diseases": [
        "infection", "infectious", "hiv", "AIDS", "virus", "viral", "bacteria", "bacterial",
        "fungi", "fungal", "parasite", "parasitic", "sepsis", "malaria", "tuberculosis",
        "TB", "influenza", "flu", "covid", "ebola", "zika", "dengue", "hepatitis", "STD",
        "sexually transmitted", "pneumonia", "MRSA"
    ],

    "Endocrinology & Metabolism": [
        "diabetes", "insulin", "metabolic", "metabolism", "thyroid", "goiter", "hyperthyroidism",
        "hypothyroidism", "cushing", "addison", "adrenal", "pituitary", "growth hormone",
        "osteoporosis", "osteopenia", "lipid", "cholesterol", "obesity", "glucose", "pcos",
        "endocrine", "hormone", "hormonal"
    ],

    "Gastroenterology": [
        "gastro", "gastrointestinal", "GI", "colitis", "crohn", "ulcerative", "hepatic",
        "liver", "pancreas", "pancreatitis", "esophagus", "esophagitis", "gastric", "stomach",
        "duodenal", "hepatitis", "cirrhosis", "IBD", "GERD", "acid reflux", "colorectal",
        "constipation", "diarrhea"
    ],

    "Respiratory": [
        "asthma", "copd", "bronch", "bronchitis", "pneumonia", "lung", "pulmonary", "respiratory",
        "emphysema", "tb", "sleep apnea", "sarcoidosis", "cystic fibrosis", "pleurisy",
        "pulmonology", "ARDS", "bronchiectasis"
    ],

    "Musculoskeletal": [
        "arthritis", "rheumatoid", "osteoarthritis", "gout", "osteoporosis", "joints",
        "joint", "muscle", "bursitis", "tendinitis", "tendon", "myopathy", "fibromyalgia",
        "scoliosis", "osteopenia", "spine", "back pain", "skeletal", "orthopedic"
    ],

    "Dermatology": [
        "dermatitis", "eczema", "psoriasis", "pemphigoid", "acne", "rosacea", "vitiligo",
        "melanoma", "skin", "cutaneous", "dermo", "onychomycosis", "pruritus", "urticaria",
        "seborrheic", "pemphigus"
    ],

    "Psychiatry": [
        "depression", "anhedonia", "ptsd", "anxiety", "bipolar", "schizophrenia", "agitation",
        "mania", "psychosis", "obsessive", "compulsive", "OCD", "addiction", "substance use",
        "suicide", "mood disorder", "sleep disorder"
    ],

    "Immunology": [
        "autoimmune", "immune", "immunology", "immunodeficiency", "lupus", "transplant",
        "graft", "GVHD", "cytokine", "inflammation", "allergy", "hypersensitivity",
        "rheumatology", "psoriatic", "crohn", "colitis"
    ],

    "Renal": [
        "renal", "kidney", "nephritis", "nephropathy", "dialysis", "transplant", "glomerular",
        "uremia", "proteinuria", "renal failure", "CKD", "acute kidney injury"
    ],

    "Rheumatology": [
        "arthritis", "rheumatoid", "spondyloarthritis", "ankylosing", "psoriatic",
        "lupus", "vasculitis", "sjogren", "gout", "fibromyalgia", "connective tissue"
    ],

    "Hematology": [
        "leukemia", "lymphoma", "myeloma", "anemia", "hemophilia", "thrombosis", "platelet",
        "erythrocyte", "hematopoietic", "stem cell", "sickle cell", "thalassemia", "bleeding"
    ],

    "Ophthalmology": [
        "eye", "ocular", "vision", "glaucoma", "cataract", "retina", "macular", "uveitis",
        "optic", "ophthalmic", "cornea", "conjunctivitis"
    ],

    "Reproductive Health": [
        "uterine", "ovarian", "prostate", "fallopian", "endometrial", "fertility", "pcos",
        "menopause", "testicular", "cervical", "vaginal", "andropause"
    ],

    "Pain Management": [
        "pain", "analgesic", "opioid", "neuropathic", "chronic pain", "headache", "migraine",
        "fibromyalgia", "arthralgia", "nociceptive", "regional anesthesia", "block"
    ]
}

def map_area(condition: str) -> str:
    """
    Maps a medical condition string to a high-level therapeutic area using keyword matching.

    Parameters:
        condition (str): Raw condition name

    Returns:
        str: Therapeutic area
    """
    cond = str(condition).lower()
    for area, keys in therapeutic_area_keywords.items():
        for kw in keys:
            if kw in cond:
                return area
    return 'Other'

def create_therapeutic_area_features(df: pd.DataFrame,therapeutic_rates: pd.Series = None) -> pd.DataFrame:
    """
    Assigns therapeutic area based on 'conditions' and calculates area-level approval rate.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'conditions' and 'outcome'

    Returns:
        pd.DataFrame: Updated DataFrame with therapeutic area features
    """
    df = df.copy()
    if 'conditions' in df.columns:
        exploded_conditions = (
            df[['nct_number', 'conditions']]
              .assign(condition=lambda d: d['conditions'].astype(str).str.split('|'))
              .explode('condition')
        )
        exploded_conditions['condition'] = exploded_conditions['condition'].str.strip()
        exploded_conditions['therapeutic_area_per_condition'] = exploded_conditions['condition'].apply(map_area)
        trial_therapeutic_area = (
            exploded_conditions
            .groupby('nct_number')['therapeutic_area_per_condition']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Other')
            .reset_index(name='therapeutic_area')
        )
        df = df.merge(trial_therapeutic_area, on='nct_number', how='left')
        df['therapeutic_area'] = df['therapeutic_area'].fillna('Other')
    if therapeutic_rates is None and 'outcome' in df.columns:
        tmp = df.copy()
        tmp['outcome_numeric'] = tmp['outcome'].map({'Approved': 1, 'Failed': 0})
        therapeutic_rates = tmp.groupby('therapeutic_area')['outcome_numeric'].mean().rename('therapeutic_area_approval_rate')
    if therapeutic_rates is not None:
        df['therapeutic_area_approval_rate'] = df['therapeutic_area'].map(therapeutic_rates).fillna(therapeutic_rates.mean())
    return df.drop(columns=['outcome_numeric'], errors='ignore')


# -----------------------------------
# Compute and store approval rates
# -----------------------------------
def compute_approval_rates(df: pd.DataFrame):
    """
    From a DataFrame with 'sponsor', 'outcome', and (after mapping) 'therapeutic_area',
    return two Series:
      - sponsor_rates: mean approved (0–1) per sponsor
      - therapeutic_rates: mean approved per therapeutic area
    """
    # numeric outcome
    tmp = df.copy()
    tmp['outcome_numeric'] = tmp['outcome'].map({'Approved': 1, 'Failed': 0})
    
    sponsor_rates = tmp.groupby('sponsor')['outcome_numeric']\
                       .mean().rename('sponsor_approval_rate')
    
    therapeutic_rates = tmp.groupby('therapeutic_area')['outcome_numeric']\
                           .mean().rename('therapeutic_area_approval_rate')
    
    return sponsor_rates, therapeutic_rates

# -----------------------------------
# Study Design Features
# -----------------------------------
def process_study_design(df: pd.DataFrame, column_name: str='study_design') -> pd.DataFrame:
    """
    Parses 'study_design' string column to extract structured features:
    - Allocation
    - Masking type/roles
    - Primary purpose
    - Composite flags

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the study design column

    Returns:
        pd.DataFrame: DataFrame with structured study design features
    """

    df = df.copy()

    # Step 1: Extract fields using regex
    # Use .str.extract with expand=True explicitly
    extracted_cols = df[column_name].str.extract(
        r'Allocation:\s*([^|]*)\|Intervention Model:\s*([^|]*)\|Masking:\s*([^|]*)\|Primary Purpose:\s*(.*)', expand=True
    )

    # Assign extracted columns
    df['Allocation'] = extracted_cols[0]
    df['Intervention Model'] = extracted_cols[1]
    df['Masking'] = extracted_cols[2]
    df['Primary Purpose'] = extracted_cols[3]

    # Step 2: Clean & Normalize
    for col in ['Allocation', 'Intervention Model', 'Masking', 'Primary Purpose']:
        df[col] = df[col].str.strip().fillna('NA').str.upper()

    # Step 3: Extract masking level
    df['masking_level'] = df['Masking'].str.extract(r'^(\w+)').fillna('NA')

    # Step 4: Binary flags for masked roles
    df['masked_participant'] = df['Masking'].str.contains('PARTICIPANT', na=False).astype(int)
    df['masked_care_provider'] = df['Masking'].str.contains('CARE_PROVIDER', na=False).astype(int)
    df['masked_investigator'] = df['Masking'].str.contains('INVESTIGATOR', na=False).astype(int)
    df['masked_outcomes_assessor'] = df['Masking'].str.contains('OUTCOMES_ASSESSOR', na=False).astype(int)

    # Step 5: Composite features
    df['is_randomized_parallel'] = (
        df['Allocation'].eq('RANDOMIZED') & df['Intervention Model'].eq('PARALLEL')
    ).astype(int)

    # Handle high masking with string match (for both encodings)
    df['high_masking'] = df['masking_level'].isin(['TRIPLE', 'QUADRUPLE']).astype(int)
    return df

# -----------------------------------
# Feature Encoding
# -----------------------------------
def encode_features(df: pd.DataFrame, cat_cols: list, label_cols: list=None):
    """
    Applies one-hot encoding to categorical columns and label encoding to label columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cat_cols (list): Columns to one-hot encode
        label_cols (list, optional): Columns to label encode

    Returns:
        Tuple[pd.DataFrame, dict]: Transformed DataFrame and encoders used
    """

    df = df.copy()
    encoders = {}
    # Label encode
    if label_cols:
        for col in label_cols:
            le = LabelEncoder()
            # Handle potential NaN values by converting to string before encoding
            df[col] = le.fit_transform(df[col].astype(str).fillna('')) # Fill NaN with empty string for consistent encoding
            encoders[col] = le
    # One-hot encode
    if cat_cols:
        # Handle potential NaN values in categorical columns before encoding
        for col in cat_cols:
             df[col] = df[col].fillna('Missing_Category') # Fill NaN with a placeholder category
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Use sparse_output instead of sparse
        arr = ohe.fit_transform(df[cat_cols].astype(str))
        ohe_cols = ohe.get_feature_names_out(cat_cols)
        # Ensure index alignment when creating the new DataFrame
        df_ohe = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), df_ohe], axis=1)
        encoders['ohe'] = ohe
    return df, encoders

# -----------------------------------
# Master Feature Engineering Pipeline
# -----------------------------------
def engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full end-to-end feature engineering pipeline combining:
    - Column name cleaning
    - Status group creation
    - Date parsing
    - Sponsor, location, and intervention processing
    - Therapeutic area assignment
    - Study design breakdown

    Parameters:
        df (pd.DataFrame): Raw trial dataset

    Returns:
        pd.DataFrame: Feature-enriched dataset
    """

    df = clean_column_names(df)
    df = create_status_group(df)
    df = create_date_features(df)
    df = create_sponsor_approval_rate(df)
    df = create_location_features(df)
    df = create_intervention_features(df)
    df = create_therapeutic_area_features(df)
    df = process_study_design(df)
    df = clean_column_names(df)
    return df

# -----------------------------------
# Test dataset Feature Engineering Pipeline
# -----------------------------------
def engineering_pipeline_test(train_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Full end-to-end feature engineering pipeline combining:
    - Column name cleaning
    - Status group creation
    - Date parsing
    - Sponsor, location, and intervention processing
    - Therapeutic area assignment
    - Study design breakdown

    Parameters:
        train_df (pd.DataFrame): Raw trial dataset
        test_df (pd.DataFrame): Raw trial dataset

    Returns:
        pd.DataFrame: Feature-enriched dataset
    """
    sponsor_rates, therapeutic_rates = compute_approval_rates(train_df)
    df = clean_column_names(df)
    df = create_status_group(df)
    df = create_date_features(df)
    df = create_sponsor_approval_rate(df, sponsor_rates)
    df = create_location_features(df)
    df = create_intervention_features(df)
    df = create_therapeutic_area_features(df, therapeutic_rates)
    df = process_study_design(df)
    df = clean_column_names(df)
    return df