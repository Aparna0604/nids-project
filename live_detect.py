"""
Day 4: Capture live network traffic and classify it as normal/attack
in real time using the trained model.

IMPORTANT (Windows):
1. Install Npcap first: https://npcap.com/#download
   (Scapy needs this to actually capture packets on Windows.)
   During install, check "Install Npcap in WinPcap API-compatible mode".
2. You MUST run Command Prompt as Administrator for sniffing to work.
   Right-click Command Prompt in the Start menu -> "Run as administrator".

Run with: python live_detect.py
Press Ctrl+C to stop.
"""
import time
import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, IP, TCP, UDP

# Load everything Day 2/3 already built.
model = joblib.load("nids_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")
target_le = encoders['target']

# A live packet doesn't carry all 41 NSL-KDD features -- many of those
# (like same_srv_rate) summarize patterns across MANY past connections.
# We approximate what we reasonably can, and for everything else we
# default to the training-set average (scaler.mean_), since StandardScaler
# centers every feature around 0 -- using the mean is the most "neutral"
# guess we can make for a feature we can't directly observe.
DEFAULT_ROW = dict(zip(feature_cols, scaler.mean_))

# A small lookup table mapping common port numbers to NSL-KDD service
# names. Real traffic uses far more ports than this; anything not listed
# falls back to 'other', which is itself a valid NSL-KDD service category.
PORT_SERVICE_MAP = {
    20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
    53: 'domain_u', 79: 'finger', 80: 'http', 110: 'pop_3', 111: 'sunrpc',
    113: 'auth', 119: 'nntp', 123: 'ntp_u', 143: 'imap4', 161: 'snmp',
    194: 'IRC', 443: 'http', 513: 'login', 514: 'shell', 3306: 'sql_net',
}

# Tracks recent connection timestamps so we can approximate NSL-KDD's
# "count" (connections to same host in last 2 sec) and "srv_count"
# (connections to same service in last 2 sec) -- this is a real, useful
# signal: a sudden spike of connections to one host is a classic
# scan/flood pattern.
recent_to_host = {}
recent_to_service = {}
WINDOW_SECONDS = 2

def update_and_count(tracker, key):
    now = time.time()
    tracker.setdefault(key, [])
    tracker[key] = [t for t in tracker[key] if now - t < WINDOW_SECONDS]
    tracker[key].append(now)
    return len(tracker[key])

def safe_encode(encoder, value, fallback='other'):
    """Encode a category, falling back gracefully if it's a value the
    encoder never saw during training (live traffic can include things
    the training data didn't)."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        try:
            return encoder.transform([fallback])[0]
        except ValueError:
            return 0

def extract_features(pkt):
    row = dict(DEFAULT_ROW)  # start from training averages

    if not pkt.haslayer(IP):
        return None

    if pkt.haslayer(TCP):
        proto, sport, dport = 'tcp', pkt[TCP].sport, pkt[TCP].dport
        flags = str(pkt[TCP].flags)
        if flags == 'S':
            flag = 'S0'
        elif 'R' in flags:
            flag = 'REJ'
        else:
            flag = 'SF'
    elif pkt.haslayer(UDP):
        proto, sport, dport, flag = 'udp', pkt[UDP].sport, pkt[UDP].dport, 'SF'
    else:
        proto, sport, dport, flag = 'icmp', 0, 0, 'SF'

    service = PORT_SERVICE_MAP.get(dport, 'other')
    dst_ip = pkt[IP].dst

    row['protocol_type'] = safe_encode(encoders['protocol_type'], proto, 'tcp')
    row['service'] = safe_encode(encoders['service'], service, 'other')
    row['flag'] = safe_encode(encoders['flag'], flag, 'SF')
    row['src_bytes'] = len(pkt)
    row['count'] = update_and_count(recent_to_host, dst_ip)
    row['srv_count'] = update_and_count(recent_to_service, (dst_ip, service))

    ordered = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    return scaler.transform(ordered), dst_ip, dport, proto

def handle_packet(pkt):
    result = extract_features(pkt)
    if result is None:
        return
    features, dst_ip, dport, proto = result
    pred = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][pred]
    label = target_le.inverse_transform([pred])[0]

    tag = "[ATTACK]" if label == "attack" else "[normal]"
    print(f"{tag} {proto.upper():4s} -> {dst_ip}:{dport}  (confidence: {confidence:.2f})")

print("Listening for live traffic... press Ctrl+C to stop.\n")
sniff(prn=handle_packet, store=False)