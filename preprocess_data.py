import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

df = pd.read_csv("KDDTrain.txt", names=columns)

df['binary_label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
print("Binary label counts:\n", df['binary_label'].value_counts())

cat_cols = ['protocol_type', 'service', 'flag']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"{col}: {len(le.classes_)} unique values encoded")

target_le = LabelEncoder()
df['binary_label_encoded'] = target_le.fit_transform(df['binary_label'])
encoders['target'] = target_le
print("Target classes:", target_le.classes_, "-> encoded as 0 and 1 respectively")

feature_cols = [c for c in columns if c != 'label']
X = df[feature_cols]
y = df['binary_label_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFinal feature matrix shape:", X_scaled.shape)

np.save("X_scaled.npy", X_scaled)
np.save("y.npy", y.to_numpy())
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print("\nSaved: X_scaled.npy, y.npy, scaler.pkl, encoders.pkl, feature_cols.pkl")
print("Day 2 complete.")