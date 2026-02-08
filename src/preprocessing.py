"""
Preprocessing module for diabetes dataset.

This module handles basic data preparation.
The approach is deliberately simple - no over-engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Count of missing values per column
        
    Note:
        This dataset is pre-cleaned, but we check anyway
        to demonstrate good practice.
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(f"Missing values:\n{missing[missing > 0]}")
    return missing


def prepare_features_target(df, target_col='Diabetes_binary'):
    """
    Separate features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        tuple: (X, y) where X is features and y is target
        
    Note:
        We keep this simple - just splitting the data.
        No feature engineering, as the dataset is already well-structured.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")
    
    return X, y


def scale_features(X_train, X_test=None):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame, optional): Test features
        
    Returns:
        tuple: Scaled training (and test if provided) features, and the scaler
        
    Note:
        Scaling is applied because K-Means is distance-based.
        We fit on training data only to avoid data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def get_class_distribution(y):
    """
    Get class distribution statistics.
    
    Args:
        y (pd.Series): Target variable
        
    Returns:
        dict: Distribution statistics
        
    Note:
        This helps us understand class imbalance,
        which is important for diabetes prediction.
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    distribution = {
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict(),
        'imbalance_ratio': counts[0] / counts[1] if len(counts) > 1 else None
    }
    
    return distribution
