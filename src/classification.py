"""
Classification module using Logistic Regression.

This module implements supervised learning for diabetes risk prediction.

Why Logistic Regression?
- Interpretable: Coefficients show feature importance
- Probabilistic: Provides risk scores, not just binary predictions
- Standard: Widely used in medical risk prediction
- Appropriate: Suitable for binary classification

Limitations acknowledged:
- Assumes linear relationship between features and log-odds
- May not capture complex non-linear patterns
- Performance depends on feature quality
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train logistic regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        random_state (int): Random seed
        
    Returns:
        LogisticRegression: Fitted model
        
    Note:
        We use default parameters (no extensive tuning).
        class_weight='balanced' addresses class imbalance.
        This is a practical choice for imbalanced medical data.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Fitted classifier
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        
    Returns:
        dict: Evaluation metrics
        
    Note:
        We focus on precision and recall because:
        - Precision: How many predicted diabetics actually have diabetes
        - Recall: How many actual diabetics we identify
        
        In healthcare, missing a diabetic patient (low recall) can be serious.
        False alarms (low precision) are less critical but still costly.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n=== Model Evaluation ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_test (pd.Series): True labels
        y_pred (np.array): Predicted labels
        save_path (str, optional): Path to save figure
        
    Note:
        Confusion matrix shows:
        - True Negatives (correctly identified non-diabetics)
        - False Positives (false alarms)
        - False Negatives (missed diabetics - most concerning)
        - True Positives (correctly identified diabetics)
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_test (pd.Series): True labels
        y_pred_proba (np.array): Predicted probabilities
        save_path (str, optional): Path to save figure
        
    Note:
        ROC curve shows trade-off between true positive rate
        and false positive rate at different thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()


def get_feature_importance(model, feature_names):
    """
    Get feature importance from logistic regression coefficients.
    
    Args:
        model: Fitted LogisticRegression model
        feature_names (list): Names of features
        
    Returns:
        pd.DataFrame: Feature importance sorted by absolute coefficient
        
    Note:
        Positive coefficients increase diabetes risk.
        Negative coefficients decrease diabetes risk.
        Magnitude indicates strength of association.
    """
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    
    coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    return coefficients
