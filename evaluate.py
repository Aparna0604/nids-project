"""
Official benchmark evaluation on KDDTest+ -- the harder, unseen test set.
This is what separates serious projects from tutorial-level ones.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Load the trained model and preprocessing objects from Day 2/3
model = joblib.load("nids_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")
target_le = encoders['target']

# Load and preprocess the official test set exactly the same way as training
df_test = pd.read_csv("KDDTest.txt", names=columns)
print("Test set shape:", df_test.shape)
print("Test label distribution:\n", df_test['label'].value_counts())

# Apply binary label
df_test['binary_label'] = df_test['label'].apply(
    lambda x: 'normal' if x == 'normal' else 'attack'
)

# Encode categorical columns using the SAME encoders from training.
# Unseen values (attack types in test not in training) get mapped to 0.
for col in ['protocol_type', 'service', 'flag']:
    le = encoders[col]
    df_test[col] = df_test[col].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )

# Encode target
y_test = target_le.transform(df_test['binary_label'])

# Scale using the SAME scaler fitted on training data
X_test = scaler.transform(df_test[feature_cols])

# Predict
y_pred = model.predict(X_test)

print("\n--- OFFICIAL KDDTest+ RESULTS ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=target_le.classes_))

print("\n--- COMPARISON SUMMARY ---")
print("Train-split accuracy (Day 3): 0.9992")
print("Official KDDTest+ accuracy:  ", round(accuracy_score(y_test, y_pred), 4))
print("\nThe drop is expected and documented in published NSL-KDD research.")
print("It reflects novel attack patterns the model was not trained on.")