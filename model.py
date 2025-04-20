import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

_DEVMODE = True
_SAMPLE_FRACTION = 0.03
MIN_IMPORTANCE_THRESHOLD = 0.02

print("--- Network Attack Detection Script ---")
print(f"Development Mode: {_DEVMODE}")
if _DEVMODE:
    print(f"Using sample fraction: {_SAMPLE_FRACTION}")
print("-" * 35)

print("1. Loading datasets...")
start_time = time.time()
try:
    train_df_full = pd.read_csv('train_net.csv')
    test_df_full = pd.read_csv('test_net.csv')
except FileNotFoundError:
    print(f"ERROR: Could not find data files.")
    print(f"Please ensure the CSV files exist at the specified paths.")
    exit()

print(f"Initial Train set size: {train_df_full.shape}")
print(f"Initial Test set size:  {test_df_full.shape}")
print(f"Loading took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

if _DEVMODE:
    print("Applying Development Mode sampling...")
    start_time = time.time()
    train_df = train_df_full.sample(frac=_SAMPLE_FRACTION, random_state=1)
    test_df = test_df_full.sample(frac=_SAMPLE_FRACTION, random_state=1)
    print(f"Sampled Train set size: {train_df.shape}")
    print(f"Sampled Test set size:  {test_df.shape}")
    print(f"Sampling took {time.time() - start_time:.2f} seconds.")
    print("-" * 35)
else:
    train_df = train_df_full
    test_df = test_df_full

print("2. Preprocessing data...")
start_time = time.time()

train_df['ANOMALY'].fillna(0, inplace=True)
test_df['ANOMALY'].fillna(0, inplace=True)
train_df['ALERT'].fillna('None', inplace=True)

revoked_columns = [
  'FLOW_ID', 'ID', 'ANALYSIS_TIMESTAMP', 'IPV4_SRC_ADDR',
  'IPV4_DST_ADDR', 'PROTOCOL_MAP', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
  'TOTAL_PKTS_EXP', 'TOTAL_BYTES_EXP',
]

cols_to_drop_for_features = ['ALERT'] + revoked_columns
original_features = train_df.drop(columns=cols_to_drop_for_features, errors='ignore').columns.tolist()

X_original = train_df[original_features].copy()
y_original = train_df['ALERT'].copy()

print(f"Preprocessing took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

print("4. Creating initial train/validation split for feature selection...")
start_time = time.time()

def split_maintain_distribution(X, y, test_size=0.2, random_state=9):
  sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
  train_indices, val_indices = next(sss.split(X, y))
  return X.iloc[train_indices], X.iloc[val_indices], y.iloc[train_indices], y.iloc[val_indices]

X_train_fs, _, y_train_fs, _ = split_maintain_distribution(X_original, y_original)

scaler_fs = StandardScaler()
x_train_scaled_fs = scaler_fs.fit_transform(X_train_fs)

print(f"Initial split and scaling took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

print("5. Performing feature selection using Random Forest...")
start_time = time.time()

rfc_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfc_fs.fit(x_train_scaled_fs, y_train_fs)

feature_importances = pd.DataFrame(
    rfc_fs.feature_importances_,
    index=X_train_fs.columns,
    columns=['importance']
).sort_values('importance', ascending=False)

print("Feature Importances:")
print(feature_importances)

COLUMNS = feature_importances[feature_importances['importance'] > MIN_IMPORTANCE_THRESHOLD].index.tolist()
print(f"\nSelected columns based on threshold > {MIN_IMPORTANCE_THRESHOLD}:")
print(COLUMNS)
print(f"Number of selected features: {len(COLUMNS)}")

plt.figure(figsize=(12, 6))
plt.xticks(rotation=-90)
sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
plt.title('Feature Importances from Random Forest')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()

print(f"Feature selection took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

print("5.5 Repreparing dataset with selected features...")
start_time = time.time()

X = train_df[COLUMNS].copy()
y = train_df['ALERT'].copy()
X_test_final = test_df[COLUMNS].copy()

X_train, X_val, y_train, y_val = split_maintain_distribution(X, y)

print(f"Final Train shape: {X_train.shape}, Final Validation shape: {X_val.shape}")
print("Train set distribution (final):")
print(y_train.value_counts(normalize=True))
print("\nValidation set distribution (final):")
print(y_val.value_counts(normalize=True))

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_validation_scaled = scaler.transform(X_val)
x_test_scaled = scaler.transform(X_test_final)

print(f"Final data preparation took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

print("7. Training MLP Classifier...")
start_time = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42,
)
print(f"MLP Parameters: {mlp.get_params()}")

mlp.fit(x_train_scaled, y_train)

print(f"MLP training took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

print("Evaluating MLP model on Validation Set...")
start_time = time.time()

predictions_val = mlp.predict(x_validation_scaled)

print("MLP Classification Report (Validation Set):")
print(classification_report(y_val, predictions_val))

print("MLP Confusion Matrix (Validation Set):")

cmat = confusion_matrix(y_val, predictions_val, labels=mlp.classes_)
df_cm = pd.DataFrame(cmat, index=mlp.classes_, columns=mlp.classes_)
print(df_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title('MLP Confusion Matrix (Validation Set)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

print(f"MLP validation took {time.time() - start_time:.2f} seconds.")
print("-" * 35)

print("Generating MLP predictions on the Test Set...")
start_time = time.time()

predictions_test = mlp.predict(x_test_scaled)

test_output_df = test_df.copy() if _DEVMODE else test_df_full.copy()

if _DEVMODE:
    predictions_test_aligned = pd.Series(predictions_test, index=test_df.index)
    test_output_df['PREDICTED_ALERT'] = predictions_test_aligned
else:
     test_output_df['PREDICTED_ALERT'] = predictions_test

print("\nTest Set Prediction Distribution:")
print(pd.Series(predictions_test).value_counts())

plt.figure(figsize=(8, 5))
fig = sns.countplot(x=predictions_test, order=mlp.classes_)
fig.set_title('MLP Predictions Distribution on the Test Set')
fig.set_xlabel('Predicted Alert Type')
fig.set_ylabel('Count')
fig.set_xticklabels(fig.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

print(f"\nMLP test prediction took {time.time() - start_time:.2f} seconds.")
print("-" * 35)
print("Script finished.")

plt.show()




import joblib
import os

MODEL_FILENAME = 'mlp_model.joblib'
SCALER_FILENAME = 'scaler.joblib'
COLUMNS_FILENAME = 'model_columns.joblib'


print(f"\nSaving trained components...")


joblib.dump(mlp, MODEL_FILENAME)
print(f"MLP model saved to {MODEL_FILENAME}")

joblib.dump(scaler, SCALER_FILENAME)
print(f"Scaler saved to {SCALER_FILENAME}")

joblib.dump(COLUMNS, COLUMNS_FILENAME)
print(f"Model columns saved to {COLUMNS_FILENAME}")

print("Saving complete.")
