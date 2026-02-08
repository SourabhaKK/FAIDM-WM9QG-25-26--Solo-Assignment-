"""
Data loading module for diabetes dataset.

This module loads the diabetes dataset from a local CSV file
to ensure reproducibility and transparency.
"""

import pandas as pd
import os


def load_diabetes_data():
    """
    Load diabetes dataset from local CSV file.
    
    Returns:
        pd.DataFrame: Diabetes dataset with health indicators
        
    Note:
        This loads data from the local CDC Diabetes Dataset.csv file.
        The dataset contains binary health indicators from BRFSS 2015.
        
        The original dataset has 3 classes (0=no diabetes, 1=prediabetes, 2=diabetes).
        For binary classification, classes 1 and 2 are combined into a single positive class.
        
        Source: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015
    """
    try:
        # Construct path to CSV file in data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'data', 'CDC Diabetes Dataset.csv')
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Convert 3-class target to binary
        # 0 = no diabetes (keep as 0)
        # 1 = prediabetes, 2 = diabetes (both become 1)
        df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
        
        # Drop the original 3-class column
        df = df.drop('Diabetes_012', axis=1)
        
        # Fix column name typo if present
        if 'HeartDiseaseorAttac' in df.columns:
            df = df.rename(columns={'HeartDiseaseorAttac': 'HeartDiseaseorAttack'})
        
        # Reorder columns to put target first
        cols = ['Diabetes_binary'] + [col for col in df.columns if col != 'Diabetes_binary']
        df = df[cols]
        
        print(f"Dataset loaded successfully from local CSV file")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Source: CDC Diabetes Dataset.csv (BRFSS 2015)")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find 'CDC Diabetes Dataset.csv' in data directory")
        print(f"Expected location: {csv_path}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def get_feature_info():
    """
    Return information about dataset features.
    
    Returns:
        dict: Feature descriptions
        
    Note:
        This is kept simple - just the essential information
        needed to understand the data.
    """
    info = {
        'target': 'Diabetes_binary',
        'description': 'Binary classification: 0 = no diabetes, 1 = diabetes or prediabetes',
        'features': [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
            'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
            'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
            'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
    }
    return info
