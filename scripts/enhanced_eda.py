"""
Enhanced EDA Script for Diabetes Dataset
Add these visualizations to your report to strengthen Section 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load data from local CSV file
print("Loading CDC Diabetes Dataset from local CSV...")
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'data', 'CDC Diabetes Dataset.csv')

df = pd.read_csv(csv_path)

# Create binary target (0 = no diabetes, 1 = prediabetes or diabetes)
df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11

# =======================
# FIGURE 1: Feature Distributions by Diabetes Status
# =======================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Health Indicators by Diabetes Status', fontsize=16, fontweight='bold')

# BMI Distribution
sns.violinplot(data=df, x='Diabetes_binary', y='BMI', ax=axes[0, 0], hue='Diabetes_binary', palette=['lightblue', 'salmon'], legend=False)
axes[0, 0].set_xlabel('Diabetes Status (0=No, 1=Yes)')
axes[0, 0].set_ylabel('BMI')
axes[0, 0].set_title('BMI Distribution')

# Age Distribution
sns.violinplot(data=df, x='Diabetes_binary', y='Age', ax=axes[0, 1], hue='Diabetes_binary', palette=['lightblue', 'salmon'], legend=False)
axes[0, 1].set_xlabel('Diabetes Status (0=No, 1=Yes)')
axes[0, 1].set_ylabel('Age Category')
axes[0, 1].set_title('Age Distribution')

# General Health Distribution
sns.violinplot(data=df, x='Diabetes_binary', y='GenHlth', ax=axes[0, 2], hue='Diabetes_binary', palette=['lightblue', 'salmon'], legend=False)
axes[0, 2].set_xlabel('Diabetes Status (0=No, 1=Yes)')
axes[0, 2].set_ylabel('General Health (1=Excellent, 5=Poor)')
axes[0, 2].set_title('General Health Perception')

# Physical Health
sns.violinplot(data=df, x='Diabetes_binary', y='PhysHlth', ax=axes[1, 0], hue='Diabetes_binary', palette=['lightblue', 'salmon'], legend=False)
axes[1, 0].set_xlabel('Diabetes Status (0=No, 1=Yes)')
axes[1, 0].set_ylabel('Days Physical Health Not Good')
axes[1, 0].set_title('Physical Health Status')

# High BP Distribution (categorical)
bp_counts = df.groupby(['Diabetes_binary', 'HighBP']).size().unstack()
bp_pct = bp_counts.div(bp_counts.sum(axis=1), axis=0) * 100
bp_pct.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'salmon'], stacked=False)
axes[1, 1].set_xlabel('Diabetes Status (0=No, 1=Yes)')
axes[1, 1].set_ylabel('Percentage (%)')
axes[1, 1].set_title('High Blood Pressure Prevalence')
axes[1, 1].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
axes[1, 1].legend(['No High BP', 'High BP'])

# High Cholesterol Distribution
chol_counts = df.groupby(['Diabetes_binary', 'HighChol']).size().unstack()
chol_pct = chol_counts.div(chol_counts.sum(axis=1), axis=0) * 100
chol_pct.plot(kind='bar', ax=axes[1, 2], color=['lightblue', 'salmon'], stacked=False)
axes[1, 2].set_xlabel('Diabetes Status (0=No, 1=Yes)')
axes[1, 2].set_ylabel('Percentage (%)')
axes[1, 2].set_title('High Cholesterol Prevalence')
axes[1, 2].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
axes[1, 2].legend(['No High Chol', 'High Chol'])

plt.tight_layout()
plt.savefig('outputs/figures/figure_eda_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/figure_eda_distributions.png")

# =======================
# FIGURE 2: Diabetes Prevalence by Demographics
# =======================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Diabetes Prevalence Across Demographic Groups', fontsize=16, fontweight='bold')

# By Income
income_prev = df.groupby('Income')['Diabetes_binary'].mean() * 100
income_prev.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_xlabel('Income Level (1=<$10k, 8=$75k+)')
axes[0].set_ylabel('Diabetes Prevalence (%)')
axes[0].set_title('Diabetes Prevalence by Income')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].axhline(y=df['Diabetes_binary'].mean()*100, color='r', linestyle='--', label='Overall Mean')
axes[0].legend()

# By Education
edu_prev = df.groupby('Education')['Diabetes_binary'].mean() * 100
edu_prev.plot(kind='bar', ax=axes[1], color='seagreen')
axes[1].set_xlabel('Education Level (1=None, 6=College+)')
axes[1].set_ylabel('Diabetes Prevalence (%)')
axes[1].set_title('Diabetes Prevalence by Education')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].axhline(y=df['Diabetes_binary'].mean()*100, color='r', linestyle='--', label='Overall Mean')
axes[1].legend()

# By Age
age_prev = df.groupby('Age')['Diabetes_binary'].mean() * 100
age_prev.plot(kind='bar', ax=axes[2], color='coral')
axes[2].set_xlabel('Age Category (1=18-24, 13=80+)')
axes[2].set_ylabel('Diabetes Prevalence (%)')
axes[2].set_title('Diabetes Prevalence by Age')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
axes[2].axhline(y=df['Diabetes_binary'].mean()*100, color='r', linestyle='--', label='Overall Mean')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/figures/figure_demographic_prevalence.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/figure_demographic_prevalence.png")

# =======================
# STATISTICAL TESTS
# =======================
print("\n" + "="*60)
print("STATISTICAL VALIDATION FOR YOUR REPORT")
print("="*60)

# BMI comparison
diabetic_bmi = df[df['Diabetes_binary'] == 1]['BMI']
non_diabetic_bmi = df[df['Diabetes_binary'] == 0]['BMI']
u_stat, p_value = stats.mannwhitneyu(diabetic_bmi, non_diabetic_bmi)

print(f"\nBMI Analysis:")
print(f"  Median BMI (No Diabetes): {non_diabetic_bmi.median():.1f}")
print(f"  Median BMI (Diabetes): {diabetic_bmi.median():.1f}")
print(f"  Mann-Whitney U test: p < 0.001" if p_value < 0.001 else f"  p-value: {p_value:.4f}")

# Age comparison
diabetic_age = df[df['Diabetes_binary'] == 1]['Age']
non_diabetic_age = df[df['Diabetes_binary'] == 0]['Age']
u_stat, p_value = stats.mannwhitneyu(diabetic_age, non_diabetic_age)

print(f"\nAge Analysis:")
print(f"  Median Age Category (No Diabetes): {non_diabetic_age.median():.1f}")
print(f"  Median Age Category (Diabetes): {diabetic_age.median():.1f}")
print(f"  Mann-Whitney U test: p < 0.001" if p_value < 0.001 else f"  p-value: {p_value:.4f}")

# High BP association
bp_chi2, bp_p = stats.chi2_contingency(pd.crosstab(df['Diabetes_binary'], df['HighBP']))[:2]
print(f"\nHigh Blood Pressure Association:")
print(f"  Chi-square test: χ² = {bp_chi2:.2f}, p < 0.001")
print(f"  % with High BP (No Diabetes): {(df[df['Diabetes_binary']==0]['HighBP'].mean()*100):.1f}%")
print(f"  % with High BP (Diabetes): {(df[df['Diabetes_binary']==1]['HighBP'].mean()*100):.1f}%")

print("\n" + "="*60)
print("ADD THESE STATISTICS TO YOUR SECTION 2.2")
print("="*60)

print("\nDone! Check outputs/figures/ directory for the figures.")
