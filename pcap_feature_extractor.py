"""
PCAP Feature Extractor using Scapy
Extracts UNSW-NB15 compatible features from PCAP files
"""

from scapy.all import rdpcap, TCP, UDP, IP
from collections import defaultdict
import pandas as pd
import numpy as np

# Exact UNSW-NB15 feature columns your model was trained on
FEATURE_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
    'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss',
    'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
    'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

# Protocol encoding matching your training data
PROTO_MAP = {
    'tcp': 6, 'udp': 17, 'icmp': 1,
    'other': 0
}

SERVICE_MAP = {
    'http': 1, 'ftp': 2, 'smtp': 3, 'ssh': 4,
    'dns': 5, 'ftp-data': 6, 'irc': 7, '-': 0, 'other': 0
}

STATE_MAP = {
    'FIN': 1, 'CON': 2, 'REQ': 3, 'RST': 4,
    'PAR': 5, 'URN': 6, 'no': 0, 'other': 0
}

PORT_SERVICE_MAP = {
    80: 'http', 443: 'http', 21: 'ftp', 20: 'ftp-data',
    25: 'smtp', 22: 'ssh', 53: 'dns', 194: 'irc'
}


def get_service(sport, dport):
    if dport in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[dport]
    if sport in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[sport]
    return '-'


def get_state(flags):
    if flags is None:
        return 'CON'
    f = str(flags)
    if 'F' in f:
        return 'FIN'
    if 'R' in f:
        return 'RST'
    return 'CON'


def extract_features_from_pcap(pcap_path):
    """
    Read a PCAP file and extract UNSW-NB15 compatible flow features.
    Returns a DataFrame ready for model prediction.
    """
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error reading PCAP: {e}")
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # Group packets into flows: key = (src_ip, dst_ip, sport, dport, proto)
    flows = defaultdict(list)

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]
        proto = ip.proto  # 6=TCP, 17=UDP, 1=ICMP

        sport, dport = 0, 0
        flags = None

        if pkt.haslayer(TCP):
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
            flags = pkt[TCP].flags

        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport

        flow_key = (ip.src, ip.dst, sport, dport, proto)
        flows[flow_key].append({
            'time': float(pkt.time),
            'size': len(pkt),
            'ttl': ip.ttl,
            'flags': flags,
            'sport': sport,
            'dport': dport,
            'proto': proto,
            'payload': len(pkt.payload)
        })

    if not flows:
        print("No IP flows found in PCAP.")
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    records = []

    for flow_key, pkts in flows.items():
        src_ip, dst_ip, sport, dport, proto_num = flow_key

        # Sort packets by time
        pkts = sorted(pkts, key=lambda x: x['time'])

        times = [p['time'] for p in pkts]
        sizes = [p['size'] for p in pkts]
        ttls  = [p['ttl']  for p in pkts]

        # Split into forward (src→dst) and reverse (dst→src) - approximate
        fwd = pkts  # all treated as forward for simplicity
        rev = []    # would need bidirectional tracking for full split

        dur = times[-1] - times[0] if len(times) > 1 else 0.0
        spkts = len(fwd)
        dpkts = len(rev) if rev else 0
        sbytes = sum(p['size'] for p in fwd)
        dbytes = sum(p['size'] for p in rev) if rev else 0

        rate = spkts / dur if dur > 0 else 0.0
        sttl = ttls[0] if ttls else 0
        dttl = ttls[-1] if ttls else 0

        sload = (sbytes * 8) / dur if dur > 0 else 0.0
        dload = (dbytes * 8) / dur if dur > 0 else 0.0

        # Inter-packet timing
        ipt_list = [times[i+1] - times[i] for i in range(len(times)-1)]
        sinpkt = np.mean(ipt_list) if ipt_list else 0.0
        dinpkt = 0.0
        sjit   = np.std(ipt_list)  if ipt_list else 0.0
        djit   = 0.0

        # TCP window sizes
        first_flags = pkts[0].get('flags')
        swin  = int(first_flags) if first_flags is not None else 0
        dwin  = 0
        stcpb = 0
        dtcpb = 0
        tcprtt  = 0.0
        synack  = 0.0
        ackdat  = 0.0

        smean = sbytes / spkts if spkts > 0 else 0.0
        dmean = dbytes / dpkts if dpkts > 0 else 0.0

        # Protocol/service/state encoding
        proto_str = {6:'tcp', 17:'udp', 1:'icmp'}.get(proto_num, 'other')
        service_str = get_service(sport, dport)
        state_str   = get_state(pkts[-1].get('flags'))

        proto_enc   = PROTO_MAP.get(proto_str, 0)
        service_enc = SERVICE_MAP.get(service_str, 0)
        state_enc   = STATE_MAP.get(state_str, 0)

        record = {
            'dur':                  round(dur, 6),
            'proto':                proto_enc,
            'service':              service_enc,
            'state':                state_enc,
            'spkts':                spkts,
            'dpkts':                dpkts,
            'sbytes':               sbytes,
            'dbytes':               dbytes,
            'rate':                 round(rate, 6),
            'sttl':                 sttl,
            'dttl':                 dttl,
            'sload':                round(sload, 6),
            'dload':                round(dload, 6),
            'sloss':                0,
            'dloss':                0,
            'sinpkt':               round(sinpkt, 6),
            'dinpkt':               round(dinpkt, 6),
            'sjit':                 round(sjit, 6),
            'djit':                 round(djit, 6),
            'swin':                 0,
            'stcpb':                0,
            'dtcpb':                0,
            'dwin':                 0,
            'tcprtt':               0.0,
            'synack':               0.0,
            'ackdat':               0.0,
            'smean':                round(smean, 6),
            'dmean':                round(dmean, 6),
            'trans_depth':          0,
            'response_body_len':    0,
            'ct_srv_src':           1,
            'ct_state_ttl':         1,
            'ct_dst_ltm':           1,
            'ct_src_dport_ltm':     1,
            'ct_dst_sport_ltm':     1,
            'ct_dst_src_ltm':       1,
            'is_ftp_login':         1 if service_str in ['ftp','ftp-data'] else 0,
            'ct_ftp_cmd':           0,
            'ct_flw_http_mthd':     1 if service_str == 'http' else 0,
            'ct_src_ltm':           1,
            'ct_srv_dst':           1,
            'is_sm_ips_ports':      1 if src_ip == dst_ip and sport == dport else 0,
        }

        records.append(record)

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)

    print(f"Extracted {len(df)} flows from PCAP.")

    return df