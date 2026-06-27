# AI-Powered Network Intrusion Detection System (NIDS)

A machine learning-based network intrusion detection system that classifies network traffic as normal or malicious in real time using a Random Forest classifier trained on the NSL-KDD dataset.

## Features
- Random Forest classifier trained on 125,973 NSL-KDD connection records
- Real-time packet capture using Scapy
- Live classification dashboard built with Streamlit
- Rigorous evaluation on both train-split and official KDDTest+ benchmark

## Results

| Evaluation | Accuracy | Attack Precision | Attack Recall |
|------------|----------|-----------------|---------------|
| Train-split (80/20) | 99.92% | 1.00 | 1.00 |
| Official KDDTest+ benchmark | 77.99% | 0.97 | 0.64 |

The drop from train-split to KDDTest+ is expected and consistent with published NSL-KDD research. KDDTest+ contains novel attack patterns not present in the training set, making it a more realistic measure of generalization.

## Architecture
## Dataset
NSL-KDD dataset — an improved version of the KDD Cup 1999 dataset, widely used in network intrusion detection research.
- Training: 125,973 records, 41 features, 22 attack types
- Test: 22,544 records (KDDTest+), includes novel attack types unseen in training

Download: https://github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection

## Setup

```bash
pip install pandas scikit-learn scapy streamlit matplotlib seaborn joblib
```

For live packet capture on Windows, install Npcap: https://npcap.com/#download

## Usage

```bash
# Step 1: Explore the dataset
python explore_data.py

# Step 2: Preprocess and clean data
python preprocess_data.py

# Step 3: Train the model
python train_model.py

# Step 4: Evaluate on official benchmark
python evaluate.py

# Step 5: Run live detection (requires admin/root)
python live_detect.py

# Step 6: Launch dashboard (requires admin/root)
streamlit run dashboard.py
```

## Limitations and Future Work
- Live packet capture approximates 9 of 41 NSL-KDD features from raw packets; features requiring full flow-state tracking (same_srv_rate, dst_host_srv_count, etc.) are defaulted to training-set means — a known train/serve skew tradeoff
- Binary classification (normal vs attack); multi-class detection of specific attack types is a natural extension
- Future improvement: replace raw packet sniffing with NetFlow/IPFIX for proper connection-level feature extraction
- KDDTest+ recall on attacks is 0.64, meaning some novel attack patterns are missed — expected given the dataset shift

## Tech Stack
Python, scikit-learn, pandas, Scapy, Streamlit, NumPy, joblib

## Author
Aparna — BTech CSE, third year
