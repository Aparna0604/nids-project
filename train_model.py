"""
Day 3: Train and evaluate the Random Forest model.
Run this with: python train_model.py
Needs preprocess_data.py to have run first (uses its saved .npy/.pkl files).
"""
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load what Day 2 already prepared -- no need to redo any preprocessing.
X = np.load("X_scaled.npy")
y = np.load("y.npy")
encoders = joblib.load("encoders.pkl")
target_le = encoders['target']  # tells us which number means 'attack' vs 'normal'

# STEP 1: Split data into training (80%) and testing (20%) sets.
# The model only ever learns from the training set. The test set is held back
# so we can check how well it performs on data it has never seen.
# stratify=y keeps the normal/attack ratio consistent in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# STEP 2: Train the Random Forest.
# n_estimators=100 means it builds 100 decision trees and combines their
# votes -- this is why it's called a "forest." n_jobs=-1 uses all CPU cores
# to speed up training. random_state=42 makes results reproducible.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# STEP 3: Evaluate on the held-out test set.
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

# The confusion matrix shows: [[true_attack_correct, attack_missed],
#                               [normal_misflagged, true_normal_correct]]
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Precision: of everything the model FLAGGED as attack, how much really was.
# Recall: of everything that WAS an attack, how much the model caught.
# These matter more than raw accuracy, especially since missing a real
# attack (low recall) is usually worse than a false alarm (low precision).
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=target_le.classes_))

# STEP 4: Check which features the model relied on most.
# This is useful for your README/interview -- it shows WHY the model
# makes its decisions, not just that it works.
feature_cols = joblib.load("feature_cols.pkl")
importances = model.feature_importances_
top10 = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:10]
print("\nTop 10 most important features:")
for name, score in top10:
    print(f"  {name}: {score:.4f}")

# STEP 5: Save the trained model so Day 4/6 (live detection, dashboard)
# can load it directly without retraining every time.
joblib.dump(model, "nids_model.pkl")
print("\nSaved trained model as nids_model.pkl")
print("Day 3 complete.")