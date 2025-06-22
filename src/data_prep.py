import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from feature_engineering import engineering_pipeline, clean_column_names

# -----------------------------------
# Load and Engineer Features
# -----------------------------------
def load_and_engineer_features(path: str) -> pd.DataFrame:
    """
    Load raw trial metadata from CSV, clean column names, and apply feature engineering.

    Parameters
    ----------
    path : str
        File path to the raw CSV containing trial metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features ready for splitting.
    """
    df = pd.read_csv(path)
    df = clean_column_names(df)
    df = engineering_pipeline(df)
    return df

# -----------------------------------
# Temporal Split
# -----------------------------------

def temporal_split(
    df: pd.DataFrame,
    date_col: str = 'start_year',
    train_end: int = 2019,
    val_end: int = 2021
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training, validation, and test sets based on a year threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with a numeric year column.
    date_col : str, optional
        Column name containing the year to split on, by default 'start_year'.
    train_end : int, optional
        Maximum year (inclusive) for the training set, by default 2019.
    val_end : int, optional
        Maximum year (inclusive) for the validation set, by default 2021.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    train = df[df[date_col] <= train_end].copy()
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test = df[df[date_col] > val_end].copy()
    return train, val, test

# -----------------------------------
# Input/Target Separation
# -----------------------------------

def get_input_target(
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str = 'outcome'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing input features and target column.
    input_cols : List[str]
        List of column names to use as features.
    target_col : str, optional
        Name of the target column to predict, by default 'outcome'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features DataFrame), y (target Series mapped to 0/1).
    """
    X = df[input_cols].copy()
    y = df[target_col].map({'Approved': 1, 'Failed': 0}).astype(int)
    return X, y

# -----------------------------------
# Identify Column Types
# -----------------------------------

def identify_column_types(
    df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical column names in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input features DataFrame.

    Returns
    -------
    Tuple[List[str], List[str]]
        num_cols: List of numeric column names.
        cat_cols: List of object (categorical) column names.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return num_cols, cat_cols

# -----------------------------------
# Imputation and Scaling
# -----------------------------------

def impute_and_scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in numeric columns and scale them to [0,1].

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature sets for training, validation, and testing.
    num_cols : List[str]
        List of numeric column names to process.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Transformed training, validation, and test feature DataFrames.
    """
    imputer = SimpleImputer(strategy='median')
    scaler = MinMaxScaler()
    # Fit on training set
    X_tr_num = pd.DataFrame(
        imputer.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )
    X_tr_num = pd.DataFrame(
        scaler.fit_transform(X_tr_num),
        columns=num_cols, index=X_train.index
    )
    # Transform val and test
    X_val_num = pd.DataFrame(
        scaler.transform(
            pd.DataFrame(imputer.transform(X_val[num_cols]), columns=num_cols)
        ), columns=num_cols, index=X_val.index
    )
    X_te_num = pd.DataFrame(
        scaler.transform(
            pd.DataFrame(imputer.transform(X_test[num_cols]), columns=num_cols)
        ), columns=num_cols, index=X_test.index
    )
    # Update original DataFrames
    X_train.update(X_tr_num)
    X_val.update(X_val_num)
    X_test.update(X_te_num)
    return X_train, X_val, X_test

# -----------------------------------
# Categorical Encoding
# -----------------------------------

def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode specified categorical columns.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature sets for training, validation, and testing.
    cat_cols : List[str]
        List of categorical column names to encode.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames with categorical columns replaced by their one-hot encodings.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # Fit on training
    arr_tr = encoder.fit_transform(X_train[cat_cols].astype(str))
    cols_ohe = encoder.get_feature_names_out(cat_cols)
    df_ohe_tr = pd.DataFrame(arr_tr, columns=cols_ohe, index=X_train.index)
    # Transform val/test
    df_ohe_val = pd.DataFrame(
        encoder.transform(X_val[cat_cols].astype(str)),
        columns=cols_ohe, index=X_val.index
    )
    df_ohe_te = pd.DataFrame(
        encoder.transform(X_test[cat_cols].astype(str)),
        columns=cols_ohe, index=X_test.index
    )
    # Concatenate and drop originals
    X_train = pd.concat([X_train.drop(columns=cat_cols), df_ohe_tr], axis=1)
    X_val = pd.concat([X_val.drop(columns=cat_cols), df_ohe_val], axis=1)
    X_test = pd.concat([X_test.drop(columns=cat_cols), df_ohe_te], axis=1)
    return X_train, X_val, X_test

# -----------------------------------
# SMOTE Resampling
# -----------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance class distribution in the training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled X_train and y_train with balanced classes.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

# -----------------------------------
# Full Pre-Modeling Pipeline
# -----------------------------------

def prepare_datasets(
    feature_path: str,
    input_cols: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end data preparation pipeline:
      - Load and engineer features
      - Temporal split
      - Separate inputs and targets
      - Identify column types
      - Impute and scale numeric features
      - Encode categorical features
      - Apply SMOTE on training set

    Parameters
    ----------
    feature_path : str
        Path to the CSV with engineered features.
    input_cols : List[str]
        List of column names to use as model inputs.

    Returns
    -------
    Dict[str, pd.DataFrame or pd.Series]
        Dictionary with keys: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
    """
    # Load and engineer
    df = load_and_engineer_features(feature_path)
    # Split
    train, val, test = temporal_split(df)
    # Inputs and targets
    X_tr, y_tr = get_input_target(train, input_cols)
    X_val, y_val = get_input_target(val, input_cols)
    X_te, y_te = get_input_target(test, input_cols)
    # Column types
    num_cols, cat_cols = identify_column_types(X_tr)
    # Impute & scale
    X_tr, X_val, X_te = impute_and_scale(X_tr, X_val, X_te, num_cols)
    # Encode categoricals
    X_tr, X_val, X_te = encode_categoricals(X_tr, X_val, X_te, cat_cols)
    # SMOTE
    X_res, y_res = apply_smote(X_tr, y_tr)
    return {
        'X_train': X_res, 'y_train': y_res,
        'X_val': X_val,  'y_val': y_val,
        'X_test': X_te,  'y_test': y_te
    }
