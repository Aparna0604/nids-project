"""
Day 6: Streamlit dashboard for the live NIDS.
"""
import warnings
warnings.filterwarnings("ignore")
import threading
import time
from collections import deque
import pandas as pd
import streamlit as st
import joblib
from scapy.all import sniff, IP, TCP, UDP

model = joblib.load("nids_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")
target_le = encoders['target']

DEFAULT_ROW = dict(zip(feature_cols, scaler.mean_))
PORT_SERVICE_MAP = {
    20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
    53: 'domain_u', 79: 'finger', 80: 'http', 110: 'pop_3', 111: 'sunrpc',
    113: 'auth', 119: 'nntp', 123: 'ntp_u', 143: 'imap4', 161: 'snmp',
    194: 'IRC', 443: 'http', 513: 'login', 514: 'shell', 3306: 'sql_net',
}
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
    try:
        return encoder.transform([value])[0]
    except ValueError:
        try:
            return encoder.transform([fallback])[0]
        except ValueError:
            return 0

def extract_features(pkt):
    row = dict(DEFAULT_ROW)
    if not pkt.haslayer(IP):
        return None
    if pkt.haslayer(TCP):
        proto, dport = 'tcp', pkt[TCP].dport
        flags = str(pkt[TCP].flags)
        flag = 'S0' if flags == 'S' else ('REJ' if 'R' in flags else 'SF')
    elif pkt.haslayer(UDP):
        proto, dport, flag = 'udp', pkt[UDP].dport, 'SF'
    else:
        proto, dport, flag = 'icmp', 0, 'SF'

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

EVENT_LOG = deque(maxlen=300)
STATS = {"normal": 0, "attack": 0}

def handle_packet(pkt):
    result = extract_features(pkt)
    if result is None:
        return
    features, dst_ip, dport, proto = result
    pred = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][pred]
    label = target_le.inverse_transform([pred])[0]

    STATS[label] += 1
    EVENT_LOG.append({
        "time": time.strftime("%H:%M:%S"),
        "status": label,
        "protocol": proto.upper(),
        "destination": f"{dst_ip}:{dport}",
        "confidence": round(float(confidence), 2),
    })

def start_sniffing():
    sniff(prn=handle_packet, store=False)

if "sniffer_started" not in st.session_state:
    thread = threading.Thread(target=start_sniffing, daemon=True)
    thread.start()
    st.session_state["sniffer_started"] = True

st.set_page_config(page_title="AI-Powered NIDS", layout="wide")
st.title("AI-Powered Network Intrusion Detection System")
st.caption("Random Forest classifier (NSL-KDD) + live packet capture via Scapy")

placeholder = st.empty()

while True:
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Normal connections", STATS["normal"])
        col2.metric("Flagged attacks", STATS["attack"])
        total = STATS["normal"] + STATS["attack"]
        rate = (STATS["attack"] / total * 100) if total else 0
        col3.metric("Attack rate", f"{rate:.1f}%")

        if EVENT_LOG:
            df = pd.DataFrame(list(EVENT_LOG)[::-1])
            st.dataframe(df, use_container_width=True, height=500)
        else:
            st.info("Waiting for traffic... browse a few sites to see activity.")

    time.sleep(1)