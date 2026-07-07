"""
Data Preprocessing Module
==========================
Handles all data loading, cleaning, feature scaling,
class balancing (SMOTE), and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.utils import setup_logging

# Initialize logger
logger = setup_logging()


# ──────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────

def load_dataset(filepath):
    """
    Load the credit card transactions dataset from a CSV file.
    
    Args:
        filepath (str): Path to the creditcard.csv file
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.ParserError: If the CSV is malformed
    """
    logger.info(f"Loading dataset from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully — Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at: {filepath}")
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please download 'creditcard.csv' from:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the 'dataset/' folder."
        )
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise


# ──────────────────────────────────────────────
#  Data Inspection
# ──────────────────────────────────────────────

def inspect_data(df):
    """
    Print comprehensive information about the dataset.
    
    Displays:
        - Shape (rows, columns)
        - Column data types
        - Missing values per column
        - Duplicate row count
        - Class distribution
        - Basic statistics
    
    Args:
        df (pd.DataFrame): The dataset to inspect
    
    Returns:
        dict: Summary statistics for programmatic use
    """
    logger.info("=" * 60)
    logger.info("DATASET INSPECTION")
    logger.info("=" * 60)
    
    # Basic shape
    logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Data types
    logger.info(f"\nData Types:\n{df.dtypes.value_counts().to_string()}")
    
    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    logger.info(f"\nTotal Missing Values: {total_missing}")
    if total_missing > 0:
        logger.info(f"Missing per column:\n{missing[missing > 0].to_string()}")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"\nDuplicate Rows: {duplicates}")
    
    # Class distribution
    class_counts = df["Class"].value_counts()
    fraud_pct = (class_counts.get(1, 0) / len(df)) * 100
    logger.info(f"\nClass Distribution:")
    logger.info(f"  Legitimate (0): {class_counts.get(0, 0):,}")
    logger.info(f"  Fraudulent (1): {class_counts.get(1, 0):,}")
    logger.info(f"  Fraud Percentage: {fraud_pct:.3f}%")
    
    # Basic statistics
    logger.info(f"\nTransaction Amount Statistics:")
    logger.info(f"  Mean:   ${df['Amount'].mean():.2f}")
    logger.info(f"  Median: ${df['Amount'].median():.2f}")
    logger.info(f"  Max:    ${df['Amount'].max():.2f}")
    logger.info(f"  Min:    ${df['Amount'].min():.2f}")
    
    logger.info("=" * 60)
    
    return {
        "shape": df.shape,
        "missing_values": total_missing,
        "duplicates": duplicates,
        "fraud_count": class_counts.get(1, 0),
        "legitimate_count": class_counts.get(0, 0),
        "fraud_percentage": fraud_pct,
    }


# ──────────────────────────────────────────────
#  Data Cleaning
# ──────────────────────────────────────────────

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing duplicates.
    
    Steps:
        1. Drop duplicate rows
        2. Fill missing numerical values with column median
        3. Verify no missing values remain
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    logger.info("Starting data cleaning...")
    initial_rows = len(df)
    
    # Step 1: Remove duplicate rows
    df = df.drop_duplicates()
    dropped_dupes = initial_rows - len(df)
    if dropped_dupes > 0:
        logger.info(f"Removed {dropped_dupes} duplicate rows")
    else:
        logger.info("No duplicate rows found")
    
    # Step 2: Handle missing values (fill with median for numeric columns)
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"Found {missing_before} missing values — filling with median")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info("Missing values handled successfully")
    else:
        logger.info("No missing values found")
    
    # Verify
    assert df.isnull().sum().sum() == 0, "Missing values still present after cleaning!"
    logger.info(f"Cleaned dataset shape: {df.shape}")
    
    return df


# ──────────────────────────────────────────────
#  Feature Scaling
# ──────────────────────────────────────────────

def scale_features(df):
    """
    Normalize the 'Time' and 'Amount' features using StandardScaler.
    
    Note: V1–V28 are already PCA-transformed and scaled.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    
    Returns:
        tuple: (scaled_df, scaler) — The scaled DataFrame and the fitted scaler
    """
    logger.info("Scaling 'Time' and 'Amount' features...")
    
    scaler = StandardScaler()
    
    # Scale only Time and Amount (V1-V28 are already PCA-scaled)
    df = df.copy()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    
    logger.info("Feature scaling completed")
    return df, scaler


# ──────────────────────────────────────────────
#  SMOTE Oversampling
# ──────────────────────────────────────────────

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    
    SMOTE generates synthetic examples of the minority class (fraud)
    to create a balanced training set.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix
        y (pd.Series or np.ndarray): Target variable
        random_state (int): Seed for reproducibility
    
    Returns:
        tuple: (X_resampled, y_resampled) — Balanced feature matrix and target
    """
    logger.info("Applying SMOTE oversampling...")
    logger.info(f"  Before SMOTE — Class 0: {sum(y == 0):,}, Class 1: {sum(y == 1):,}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    logger.info(f"  After SMOTE  — Class 0: {sum(y_resampled == 0):,}, Class 1: {sum(y_resampled == 1):,}")
    logger.info("SMOTE balancing completed")
    
    return X_resampled, y_resampled


# ──────────────────────────────────────────────
#  Train/Test Split
# ──────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix
        y (pd.Series or np.ndarray): Target variable
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data — Test size: {test_size*100:.0f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class proportions
    )
    
    logger.info(f"  Training set: {X_train.shape[0]:,} samples")
    logger.info(f"  Testing set:  {X_test.shape[0]:,} samples")
    
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
#  Full Preprocessing Pipeline
# ──────────────────────────────────────────────

def preprocess_pipeline(filepath):
    """
    Execute the complete preprocessing pipeline end-to-end.
    
    Steps:
        1. Load dataset
        2. Inspect data
        3. Clean data (duplicates, missing values)
        4. Scale features (Time, Amount)
        5. Separate features (X) and target (y)
        6. Split into train/test
        7. Apply SMOTE to training data only
    
    Args:
        filepath (str): Path to creditcard.csv
    
    Returns:
        dict: Contains X_train, X_test, y_train, y_test, scaler, 
              original_df, and inspection summary
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load
    df = load_dataset(filepath)
    
    # 2. Inspect
    summary = inspect_data(df)
    
    # 3. Clean
    df_clean = clean_data(df)
    
    # 4. Scale
    df_scaled, scaler = scale_features(df_clean)
    
    # 5. Separate features and target
    X = df_scaled.drop("Class", axis=1)
    y = df_scaled["Class"]
    
    # 6. Split (before SMOTE to avoid data leakage)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 7. SMOTE on training data only
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return {
        "X_train": X_train_balanced,
        "X_test": X_test,
        "y_train": y_train_balanced,
        "y_test": y_test,
        "scaler": scaler,
        "original_df": df,
        "cleaned_df": df_clean,
        "feature_names": X.columns.tolist(),
        "summary": summary,
    }
