#!/usr/bin/env python3
"""
Logistic Regression
Testing donor engagement diversity as predictor of planned giving
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

df = pd.read_csv('individual_features_engineered.csv')

features = [
    'unique_campaigns',
    'unique_designations',
    'unique_opportunity_types',
    'unique_credit_types'
]

target = 'household_has_pg'

X = df[features]
y = df[target]

print(f"Total observations: {len(df):,}")
print(f"PG = 0: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"PG = 1: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")

print("\nMissing values:")
for feat in features:
    missing = df[feat].isna().sum()
    missing_pct = (missing / len(df)) * 100
    print(f"  {feat}: {missing:,} ({missing_pct:.2f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("\n" + "="*80)
print("FEATURE COEFFICIENTS")
print("="*80)

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nIntercept: {model.intercept_[0]:.6f}\n")
for _, row in coef_df.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"{row['Feature']:40s} | {row['Coefficient']:10.6f} {direction}")

y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n" + "="*80)
print("PERFORMANCE")
print("="*80)
print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"Baseline: {(y_test == 1).sum() / len(y_test):.4f}")
print(f"Lift: {pr_auc / ((y_test == 1).sum() / len(y_test)):.2f}x")

coef_sorted = coef_df.sort_values('Coefficient')

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

colors = ['#d62728' if c > 0 else '#1f77b4' for c in coef_sorted['Coefficient']]
bars = plt.barh(coef_sorted['Feature'], coef_sorted['Coefficient'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

for i, (bar, coef) in enumerate(zip(bars, coef_sorted['Coefficient'])):
    x_pos = coef + (0.01 if coef > 0 else -0.01)
    ha = 'left' if coef > 0 else 'right'
    plt.text(x_pos, i, f'{coef:.6f}', 
             va='center', ha=ha, fontsize=11, fontweight='bold')

plt.xlabel('Coefficient Value', fontsize=13, fontweight='bold')
plt.title('Engagement Breadth - Logistic Regression', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)

legend_elements = [
    Patch(facecolor='#d62728', label='Positive'),
    Patch(facecolor='#1f77b4', label='Negative')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('engagement_breadth_coefficients.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: engagement_breadth_coefficients.png")
plt.show()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nFeatures: {len(features)}")
print(f"Test samples: {len(X_test):,}")
print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
print("\nTop features:")
for i, (_, row) in enumerate(coef_df.iterrows(), 1):
    print(f"{i}. {row['Feature']} ({row['Coefficient']:.6f})")