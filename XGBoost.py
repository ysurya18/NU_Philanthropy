
#!/usr/bin/env python3
"""
XGBoost Model for Planned Giving Prediction

@author: surya
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('individual_features_engineered.csv')

df['log_total_giving_squared'] = df['log_total_giving'] ** 2
df['engagement_score_squared'] = df['engagement_score'] ** 2
df['us_resident_X_engagement'] = df['is_us_resident'] * df['engagement_score']
df['male_X_engagement'] = df['is_male'] * df['engagement_score']
df['friend_X_engagement'] = df['is_friend'] * df['engagement_score']
df['spouse_X_alumni'] = df['has_spouse'] * df['is_alumni']
df['max_to_avg_ratio'] = df['max_gift'] / df['avg_gift_size']
df['max_to_avg_ratio'] = df['max_to_avg_ratio'].replace([np.inf, -np.inf], np.nan)
df['log_total_giving_x_engagement'] = df['log_total_giving'] * df['engagement_score']


features = [
    'is_alumni',
    'gift_cv',
    'years_since_last_gift',
    'has_masters',
    'avg_annual_giving',
    'age',
    'log_total_giving_squared',
    'engagement_score_squared',
    'us_resident_X_engagement',
    'male_X_engagement',
    'friend_X_engagement',
    'spouse_X_alumni',
    'max_to_avg_ratio',
    'log_total_giving_x_engagement'
]

target = 'household_has_pg'

X = df[features]
y = df[target]

print(f"\nObservations: {len(df):,}")
print(f"PG = 0: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"PG = 1: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")
print(f"Age missing: {df['age'].isna().sum():,} ({df['age'].isna().sum()/len(df)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'max_depth': 6,
    'learning_rate': 0.03,
    'n_estimators': 1000,
    'scale_pos_weight': scale_pos_weight,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'enable_categorical': False,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}



model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)



y_pred_proba = model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"\nPR-AUC: {pr_auc:.4f}")
print(f"Baseline: {(y_test == 1).sum() / len(y_test):.4f}")
print(f"Lift: {pr_auc / ((y_test == 1).sum() / len(y_test)):.2f}x")

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for _, row in importance_df.iterrows():
    print(f"{row['Feature']:45s} | {row['Importance']:.6f}")

results_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted_proba': y_pred_proba
}).sort_values('predicted_proba', ascending=False).reset_index(drop=True)

percentiles = [1, 5, 10, 25, 50, 75, 100]
capture_rates = []
total_pg = results_df['actual'].sum()

print("\nCapture Rate by Percentile:")
for p in percentiles:
    n_rows = int(len(results_df) * (p / 100))
    pg_captured = results_df.head(n_rows)['actual'].sum()
    capture_rate = (pg_captured / total_pg) * 100
    capture_rates.append(capture_rate)
    print(f"Top {p:3d}%: {pg_captured:4d}/{total_pg:4d} donors ({capture_rate:.1f}%)")

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

bars = plt.bar(range(len(percentiles)), capture_rates, color='steelblue', 
               alpha=0.8, edgecolor='black', linewidth=1.5)

plt.xlabel('Top Percentile', fontsize=14, fontweight='bold')
plt.ylabel('% of All PG Donors Captured', fontsize=14, fontweight='bold')
plt.title('XGBoost - PG Capture Rate', fontsize=16, fontweight='bold')
plt.xticks(range(len(percentiles)), [f'Top {p}%' for p in percentiles])
plt.ylim(0, 105)

for bar, rate in zip(bars, capture_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_capture_rate.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: xgboost_capture_rate.png")
plt.show()

print(f"\nTop 5 Features:")
for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    print(f"{i}. {row['Feature']} ({row['Importance']:.6f})")

print(f"\nTop 10% captures {capture_rates[2]:.1f}% of all planned giving donors")