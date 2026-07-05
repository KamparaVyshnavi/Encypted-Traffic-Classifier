"""
===============================================================================
Encrypted Traffic Classifier Dashboard
===============================================================================

Real-time dashboard for the Encrypted Traffic Classifier.

Responsibilities
----------------
1. Select capture interface
2. Start / Stop live capture
3. Display live statistics
4. Visualize classified traffic
5. Display latest prediction
6. Show Multi-Exit CNN performance
"""

import threading
import time

import pandas as pd
import streamlit as st

from main import EncryptedTrafficClassifier


# =============================================================================
# Streamlit Configuration
# =============================================================================

st.set_page_config(

    page_title="Encrypted Traffic Classifier",

    page_icon="🛡️",

    layout="wide",
)


# =============================================================================
# Session State
# =============================================================================

if "classifier" not in st.session_state:

    st.session_state.classifier = None

if "capture_running" not in st.session_state:

    st.session_state.capture_running = False

if "capture_thread" not in st.session_state:

    st.session_state.capture_thread = None


# =============================================================================
# Header
# =============================================================================

st.title("🛡️ Encrypted Traffic Classifier")

st.caption(
    "Real-Time Encrypted Network Traffic Classification using Multi-Exit CNN"
)

st.divider()


# =============================================================================
# Interface Discovery
# =============================================================================

temp_classifier = EncryptedTrafficClassifier()

interfaces = (
    temp_classifier
    .interface_manager
    .get_available_interfaces()
)

selected_interface = st.selectbox(

    "Select Network Interface",

    interfaces,
)


# =============================================================================
# Capture Controls
# =============================================================================

col1, col2 = st.columns(2)

with col1:

    start_button = st.button(

        "▶ Start Capture",

        width="stretch",
    )

with col2:

    stop_button = st.button(

        "■ Stop Capture",

        width="stretch",
    )


# =============================================================================
# Start Capture
# =============================================================================

if (

    start_button

    and

    not st.session_state.capture_running

):

    classifier = EncryptedTrafficClassifier()

    capture_thread = threading.Thread(

        target=classifier.start,

        args=(selected_interface,),

        daemon=True,

    )

    capture_thread.start()

    st.session_state.classifier = classifier

    st.session_state.capture_thread = capture_thread

    st.session_state.capture_running = True


# =============================================================================
# Stop Capture
# =============================================================================

if (

    stop_button

    and

    st.session_state.capture_running

):

    st.session_state.classifier.stop()

    st.session_state.capture_running = False


# =============================================================================
# Capture Status
# =============================================================================

if st.session_state.capture_running:

    st.success("🟢 Live Capture Running")

else:

    st.warning("🟡 Capture Not Running")


st.divider()

# =============================================================================
# Live Statistics
# =============================================================================

if st.session_state.capture_running:

    classifier = st.session_state.classifier

    stats = classifier.get_statistics()

else:

    stats = {

        "packet_count": 0,

        "active_flows": 0,

        "completed_flows": 0,

        "class_counter": {},

        "exit_counter": {},

        "average_latency": 0.0,

        "average_packets": 0.0,

        "latency_saved": 0.0,

        "latest_prediction": None,

    }


# =============================================================================
# System Status
# =============================================================================

st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Packets Captured",

        stats["packet_count"],

    )

with col2:

    st.metric(

        "Active Flows",

        stats["active_flows"],

    )

with col3:

    st.metric(

        "Completed Flows",

        stats["completed_flows"],

    )

with col4:

    latest = stats["latest_prediction"]

    if latest is None:

        st.metric(

            "Latest Prediction",

            "--",

        )

    else:

        st.metric(

            "Latest Prediction",

            latest["prediction"],

        )

st.divider()


# =============================================================================
# Traffic Distribution
# =============================================================================

left, right = st.columns(2)

with left:

    st.subheader("Traffic Distribution")

    traffic = stats["class_counter"]

    if traffic:

        traffic_df = pd.DataFrame(

            {

                "Traffic": list(

                    traffic.keys()

                ),

                "Flows": list(

                    traffic.values()

                ),

            }

        )

        st.bar_chart(

            traffic_df,

            x="Traffic",

            y="Flows",

            width="stretch",

        )

    else:

        st.info(

            "Waiting for classified traffic..."

        )


with right:

    st.subheader("Early Exit Usage")

    exits = stats["exit_counter"]

    if exits:

        exit_df = pd.DataFrame(

            {

                "Exit": list(

                    exits.keys()

                ),

                "Count": list(

                    exits.values()

                ),

            }

        )

        st.bar_chart(

            exit_df,

            x="Exit",

            y="Count",

            width="stretch",

        )

    else:

        st.info(

            "No exit statistics available."

        )


st.divider()

# =============================================================================
# Latest Classification
# =============================================================================

st.subheader("Latest Classification")

latest = stats["latest_prediction"]

if latest is None:

    st.info("Waiting for first prediction...")

else:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Prediction",

            latest["prediction"],

        )

    with col2:

        st.metric(

            "Confidence",

            f"{latest['confidence']:.2%}",

        )

    with col3:

        st.metric(

            "Exit Used",

            latest["exit"],

        )


st.divider()


# =============================================================================
# Performance
# =============================================================================

st.subheader("Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Average Latency",

        f"{stats['average_latency'] * 1000:.3f} ms",

    )

with col2:

    st.metric(

        "Average Packets Used",

        f"{stats['average_packets']:.2f} / 20",

    )

with col3:

    st.metric(

        "Computation Saved",

        f"{stats['latency_saved']:.2f}%",

    )


st.divider()


# =============================================================================
# Traffic Summary
# =============================================================================

st.subheader("Traffic Summary")

traffic = stats["class_counter"]

if traffic:

    total = sum(

        traffic.values()

    )

    rows = []

    for label, count in traffic.items():

        rows.append(

            {

                "Traffic Type": label,

                "Flows": count,

                "Percentage": round(

                    count * 100 / total,

                    2,

                ),

            }

        )

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(

        "Flows",

        ascending=False,

    )

    st.dataframe(

        summary_df,

        width="stretch",

        hide_index=True,

    )

else:

    st.info(

        "No classified traffic yet."

    )


st.divider()


# =============================================================================
# Footer
# =============================================================================

st.caption(

    "Encrypted Traffic Classifier | "

    "Multi-Exit CNN | "

    "Handshake-Based Temporal Normalization"

)
# =============================================================================
# Auto Refresh
# =============================================================================

if st.session_state.capture_running:

    time.sleep(1)

    st.rerun()