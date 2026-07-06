# Encrypted Traffic Profiler

### Adaptive Real-Time Encrypted Traffic Behavioral Classification System

A real-time deep learning framework for classifying encrypted network traffic using **behavioral sequence analysis** — without ever inspecting packet payloads.

> Author: **Vyshnavi Kampara**

---

## Overview

Modern internet traffic is overwhelmingly encrypted (HTTPS, TLS 1.3, VPN tunnels, QUIC), which makes traditional Deep Packet Inspection (DPI) ineffective — encrypted payloads simply can't be read. However, encrypted traffic still leaks **behavioral metadata**: packet sizes, directional flow patterns, burst structure, and timing rhythms.

This project uses that observable behavior to classify traffic in real time — capturing live packet streams, grouping them into bidirectional flows, building temporal packet sequences, and feeding them into a **Temporal Convolutional Neural Network (Temporal CNN)** that learns application-specific patterns directly from traffic dynamics.

**Classified categories:**
- Streaming
- Chat
- File Transfer
- Email
- VoIP
- P2P
All classification is **payload-independent** — only metadata is ever touched.

---

## Motivation

Existing encrypted-traffic classifiers run into two recurring problems in the real world:

### 1. Domain Shift Across Network Environments
Most models are trained on static `.pcap` datasets captured under controlled lab conditions. When deployed on real interfaces — corporate Wi-Fi, mobile hotspots, Ethernet, VPN tunnels — raw timing behavior shifts due to differing latency and bandwidth, and models trained on raw timing fail to generalize.

### 2. Rigid Early-Classification Windows
Many early-traffic classifiers require a fixed number of packets before they can make a prediction, even when a flow is obvious within the first few packets. This wastes computation and adds unnecessary latency.

---

## Novel Contribution: Adaptive Dual-Layer Framework

### Layer 1 — Handshake-Based Temporal Normalization

Instead of feeding fragile raw inter-arrival timings into the model, the system first computes the connection's handshake latency as a baseline.Every subsequent packet timing gap is normalized relative to that baseline.This turns raw, environment-sensitive timing values into **environment-relative** behavioral metrics — the model learns *application communication rhythm* instead of *physical network speed*, which improves generalization across Wi-Fi, Ethernet, hotspots, VPNs, and varying latency conditions.

### Layer 2 — Multi-Exit Temporal CNN

The normalized packet sequence is processed by a 1D dilated Temporal CNN with multiple early-exit classification heads, with inference checkpoints at:

```
Packet 5 → Packet 15 → Packet 30
```

If prediction confidence at an early checkpoint exceeds an entropy threshold, inference **exits immediately**. Otherwise, the sequence flows deeper for final classification.

This gives adaptive inference latency, lower compute cost per flow, and flow-specific dynamic processing — instead of rigid fixed-window evaluation.

---

## System Architecture

```
Network Interface
      ↓
Packet Capture Engine
      ↓
Flow Generation Engine
      ↓
Handshake-Based Normalization Layer
      ↓
Sequence Construction Layer
      ↓
Multi-Exit Temporal CNN
      ↓
Backend Services
      ↓
Database + Dashboard
```

---

## Core Components

| Component | Description |
|---|---|
| **Packet Capture Engine** | Captures live traffic from Wi-Fi/Ethernet/VPN interfaces using PyShark, Scapy, and libpcap/Npcap. Extracts only metadata — packet size, timestamp, direction, ports, protocol. |
| **Flow Generation Engine** | Groups packets into bidirectional flows via 5-tuple identification (src IP, dst IP, src port, dst port, protocol), preserving order, direction, and burst behavior. |
| **Sequence Construction Layer** | Converts each flow into a fixed-length temporal sequence of `[packet_size, direction, relative_IAT]`, normalized into ML-ready tensors. |
| **Multi-Exit Temporal CNN** | Applies 1D dilated convolutions to learn burst structure, communication rhythm, directional asymmetry, and timing motifs — with no manual feature engineering. |
| **Backend Services** | FastAPI-based async backend handling REST APIs, WebSocket streaming, active flow state, and orchestration. |
| **Database Layer** | PostgreSQL stores prediction history, flow metadata, traffic statistics, and system metrics for monitoring and analysis. |
| **Dashboard** | Real-time visualization of classified traffic, active flows, bandwidth usage, prediction timelines, traffic distribution, and early-exit stats. |

---

## Key Engineering Characteristics

| Characteristic | Purpose |
|---|---|
| Streaming-Oriented | Real-time live processing |
| Asynchronous Pipeline | Prevents packet drops |
| Modular Design | Scalability and maintainability |
| Payload-Independent | Works purely on encrypted traffic metadata |
| Flow-Based Analysis | Captures behavioral intelligence |
| Adaptive Inference | Dynamic latency optimization |
| Environment-Aware Normalization | Cross-network stability |

---

## Datasets

| Purpose | Dataset |
|---|---|
| Baseline training | [ISCXVPN2016 — VPN-nonVPN Traffic Dataset](https://www.unb.ca/cic/datasets/vpn.html) (Canadian Institute for Cybersecurity) |
| Cross-network evaluation | [VNAT — VPN/Non-VPN Network Application Traffic Dataset](https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat) (MIT Lincoln Laboratory) |

Using a separate dataset for cross-network evaluation is intentional — it directly tests the framework's core claim: that handshake-normalized features generalize better across environments than raw-timing features.

---

## Tech Stack

- **Language:** Python
- **Packet Capture:** PyShark, Scapy, libpcap / Npcap
- **Deep Learning:** PyTorch (Temporal CNN, multi-exit architecture)
- **Database:** PostgreSQL
- **Dashboards:** two separate front-ends are provided (see below)

---

## Dashboards

This project ships **two independent dashboard implementations** on top of the same classification pipeline — pick whichever fits your workflow.

### 1. Streamlit Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```
- Built with **Streamlit** — Python-only, no separate frontend code required.
- Fastest way to get a working UI directly from the classifier's Python objects.

### 2. FastAPI + Live Web Dashboard
```bash
python dashboard_server.py
```
Then open **http://127.0.0.1:8000** in your browser (don't open `dashboard.html` directly as a file — it needs to be served by the FastAPI app for the API calls to work).

- **Backend:** FastAPI + Uvicorn — exposes REST endpoints (`/api/interfaces`, `/api/start`, `/api/stop`, `/api/stats`) and serves the static dashboard page.
- **Validation:** Pydantic models for request bodies.
- **Concurrency:** Runs packet capture in a background Python `threading.Thread` so the web server stays responsive while the sniffer loop runs.
- **Frontend:** Plain HTML/CSS/JavaScript (no framework) — a single self-contained `dashboard.html`.
- **Charts:** [Chart.js](https://www.chartjs.org/) (loaded from a local `/static` copy first, falling back to a CDN) renders the two donut charts — Traffic Distribution and Early-Exit Usage.
- **Live updates:** the frontend polls `/api/stats` every second via `fetch()` and re-renders stats, charts, and the recent-predictions table — no WebSockets currently used in this implementation.
- Wraps the real `EncryptedTrafficClassifier` from `main.py` unmodified, adding only a rolling prediction history for the "Recent Predictions" table.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Npcap (Windows) or libpcap (Linux/macOS) installed for packet capture
- PostgreSQL instance (local or remote)
- Administrator/root privileges (required for live packet capture)

### Installation

```bash
git https://github.com/KamparaVyshnavi/Encypted-Traffic-Profiler.git
cd encrypted-traffic-profiler
pip install -r requirements.txt
```

### Configuration
Update connection strings, network interface names, and model paths in `utils/config.py` before running.

### Running the Pipeline

**1. Capture & classify live traffic:**
```bash
python main.py
```

**2. Train a model from processed sequences:**
```bash
python model/train.py
```

**3. Run cross-network evaluation:**
```bash
python evaluation/cross_network_test.py
```

**4. Launch a dashboard (choose one):**
```bash
# Option A — Streamlit dashboard
streamlit run dashboard/streamlit_app.py

# Option B — FastAPI + web dashboard (opens http://127.0.0.1:8000 automatically)
python dashboard_server.py
```
See the [Dashboards](#dashboards) section above for details on each.

---

## License

**© 2026 Vyshnavi Kampara. All Rights Reserved.**

---

## Author

**Vyshnavi Kampara**
Encrypted Traffic Profiler (Adaptive Real-Time Encrypted Traffic Behavioral Classification System)