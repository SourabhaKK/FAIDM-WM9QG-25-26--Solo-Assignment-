"""
STATISTICS GENERATOR FOR REPORT IMPROVEMENTS
Run this script to get all the numbers you need to fill into the LaTeX document
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import os
import warnings
warnings.filterwarnings('ignore')

# Load data from local CSV file
print("Loading CDC Diabetes Dataset from local CSV...")
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'data', 'CDC Diabetes Dataset.csv')

df = pd.read_csv(csv_path)
df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("="*80)
print("STATISTICS FOR YOUR LATEX REPORT")
print("="*80)

# ============================================================================
# PART 1: CLUSTER INTERPRETATION STATISTICS
# ============================================================================
print("\n" + "="*80)
print("PART 1: CLUSTER PROFILING (For Section 3.4)")
print("="*80)

# Prepare data for clustering
X_features = df.drop(['Diabetes_012', 'Diabetes_binary'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Profile each cluster
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    cluster_pct = (len(cluster_data) / len(df)) * 100
    diabetes_prev = cluster_data['Diabetes_binary'].mean() * 100
    median_bmi = cluster_data['BMI'].median()
    median_age = cluster_data['Age'].median()
    highbp_pct = cluster_data['HighBP'].mean() * 100
    highchol_pct = cluster_data['HighChol'].mean() * 100
    
    print(f"\n**CLUSTER {cluster}:**")
    print(f"  Population: {cluster_pct:.1f}% of total")
    print(f"  Diabetes Prevalence: {diabetes_prev:.1f}%")
    print(f"  Median BMI: {median_bmi:.1f}")
    print(f"  Median Age Category: {median_age:.1f}")
    print(f"  High BP Prevalence: {highbp_pct:.1f}%")
    print(f"  High Cholesterol Prevalence: {highchol_pct:.1f}%")
    print(f"  Median GenHlth: {cluster_data['GenHlth'].median():.1f}")

# Determine risk levels based on diabetes prevalence
cluster_stats = []
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    diabetes_prev = cluster_data['Diabetes_binary'].mean() * 100
    cluster_stats.append((cluster, diabetes_prev))

cluster_stats.sort(key=lambda x: x[1])  # Sort by diabetes prevalence
print("\n**RISK STRATIFICATION:**")
print(f"  Low Risk: Cluster {cluster_stats[0][0]} ({cluster_stats[0][1]:.1f}% diabetes prevalence)")
print(f"  Moderate Risk: Cluster {cluster_stats[1][0]} ({cluster_stats[1][1]:.1f}% diabetes prevalence)")
print(f"  High Risk: Cluster {cluster_stats[2][0]} ({cluster_stats[2][1]:.1f}% diabetes prevalence)")
print(f"  Risk Ratio (High/Low): {cluster_stats[2][1]/cluster_stats[0][1]:.1f}x")

# ============================================================================
# PART 2: FAIRNESS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 2: FAIRNESS ANALYSIS (For Section 4.8)")
print("="*80)

# Train the model
X = df.drop(['Diabetes_012', 'Diabetes_binary', 'Cluster'], axis=1)
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Create test dataframe with original features
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['y_true'] = y_test.values
test_df = test_df.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# Function to evaluate on subgroup
def evaluate_subgroup(mask, name):
    if mask.sum() == 0:
        return None
    
    X_sub = X_test_scaled[mask]
    y_sub = y_test_reset[mask]
    
    y_pred_proba = model.predict_proba(X_sub)[:, 1]
    y_pred = model.predict(X_sub)
    
    try:
        auc = roc_auc_score(y_sub, y_pred_proba)
        recall = recall_score(y_sub, y_pred)
        precision = precision_score(y_sub, y_pred)
        return auc, recall, precision
    except:
        return None

print("\n**SUBGROUP PERFORMANCE:**")

# Income groups
low_income = test_df['Income'] <= 3
high_income = test_df['Income'] >= 6

result = evaluate_subgroup(low_income, "Low Income")
if result:
    print(f"\nLow Income (1-3): AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

result = evaluate_subgroup(high_income, "High Income")
if result:
    print(f"High Income (6-8): AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

# Education groups
low_edu = test_df['Education'] <= 3
high_edu = test_df['Education'] >= 5

result = evaluate_subgroup(low_edu, "Low Education")
if result:
    print(f"\nLow Education (1-3): AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

result = evaluate_subgroup(high_edu, "High Education")
if result:
    print(f"High Education (5-6): AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

# Sex groups
female = test_df['Sex'] == 0
male = test_df['Sex'] == 1

result = evaluate_subgroup(female, "Female")
if result:
    print(f"\nFemale: AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

result = evaluate_subgroup(male, "Male")
if result:
    print(f"Male: AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

# Age groups
young = test_df['Age'] <= 6
old = test_df['Age'] >= 10

result = evaluate_subgroup(young, "Young")
if result:
    print(f"\nYoung (≤6): AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

result = evaluate_subgroup(old, "Old")
if result:
    print(f"Old (≥10): AUC={result[0]:.3f}, Recall={result[1]:.3f}, Precision={result[2]:.3f}")

# ============================================================================
# PART 3: LATEX TABLE GENERATOR
# ============================================================================
print("\n" + "="*80)
print("PART 3: COPY-PASTE LATEX CODE FOR FAIRNESS TABLE")
print("="*80)

print("""
\\begin{table}[h]
\\centering
\\caption{Model Performance by Demographic Subgroups}
\\begin{tabular}{lccc}
\\hline
\\textbf{Subgroup} & \\textbf{ROC-AUC} & \\textbf{Recall} & \\textbf{Precision} \\\\
\\hline
""")

# Re-calculate and print in LaTeX format
subgroups = [
    (low_income, "Low Income (1-3)"),
    (high_income, "High Income (6-8)"),
    (low_edu, "Low Education (1-3)"),
    (high_edu, "High Education (5-6)"),
    (female, "Female"),
    (male, "Male"),
]

for mask, name in subgroups:
    result = evaluate_subgroup(mask, name)
    if result:
        print(f"{name} & {result[0]:.2f} & {result[1]:.2f} & {result[2]:.2f} \\\\")

print("""\\hline
\\end{tabular}
\\label{tab:fairness}
\\end{table}
""")

# ============================================================================
# PART 4: COPY-PASTE TEXT FOR CLUSTER INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("PART 4: COPY-PASTE TEXT FOR CLUSTER INTERPRETATION (Section 3.4)")
print("="*80)

# Re-sort to get low/medium/high
cluster_stats.sort(key=lambda x: x[1])

for i, (cluster, diabetes_prev) in enumerate(cluster_stats):
    cluster_data = df[df['Cluster'] == cluster]
    cluster_pct = (len(cluster_data) / len(df)) * 100
    median_bmi = cluster_data['BMI'].median()
    median_age = cluster_data['Age'].median()
    highbp_pct = cluster_data['HighBP'].mean() * 100
    highchol_pct = cluster_data['HighChol'].mean() * 100
    
    risk_level = ["Low Risk", "Moderate Risk", "High Risk"][i]
    
    print(f"\n\\textbf{{Cluster {cluster} ({risk_level}, {cluster_pct:.0f}\\% of population):}} ")
    print(f"Median BMI {median_bmi:.1f}, age category {median_age:.0f}, ", end="")
    print(f"comorbidity prevalence {highbp_pct:.0f}\\% high BP, {highchol_pct:.0f}\\% high cholesterol. ")
    print(f"Diabetes prevalence: {diabetes_prev:.1f}\\%. ")
    
    if i == 0:
        print("\\textit{Public health implication:} Preventive education and lifestyle interventions sufficient.")
    elif i == 1:
        print("\\textit{Public health implication:} Targeted screening programs and early intervention warranted.")
    else:
        print("\\textit{Public health implication:} Intensive screening, clinical follow-up, and aggressive risk factor management required.")

print("\n" + "="*80)
print("DONE! Copy the sections above into your LaTeX document.")
print("="*80)
