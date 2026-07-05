"""
===============================================================================
Encrypted Traffic Classifier Dashboard
===============================================================================

Real-time dashboard for the encrypted traffic classifier.

This dashboard displays

• Traffic Distribution
• Current Prediction
• Early Exit Usage
• Performance Metrics
• Session Statistics
• Event Log

The dashboard does NOT perform packet capture.

main.py supplies all live statistics.
"""

from collections import defaultdict
import time

import pandas as pd
import streamlit as st


class Dashboard:

    def __init__(self):

        st.set_page_config(

            page_title="Encrypted Traffic Classifier",

            page_icon="🛡️",

            layout="wide",
        )

        self.start_time = time.time()

        self.class_counts = defaultdict(int)

        self.exit_counts = defaultdict(int)

        self.logs = []

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    def initialize(self):

        st.title("🛡️ Encrypted Traffic Classifier")

        st.caption(
            "Real-Time Encrypted Traffic Analysis Dashboard"
        )

        self.header = st.empty()

        col1, col2 = st.columns([2, 1])

        with col1:

            self.traffic_chart = st.empty()

            self.class_table = st.empty()

        with col2:

            self.current_prediction = st.empty()

            self.performance = st.empty()

            self.exit_usage = st.empty()

        st.divider()

        self.event_log = st.empty()
        # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    def update(

        self,

        prediction,

        confidence,

        exit_used,

        inference_time,

        average_packets,

        latency_saved,

        packets,

        flows,

    ):

        self.class_counts[prediction] += 1

        self.exit_counts[exit_used] += 1

        self.logs.insert(

            0,

            {

                "Time": time.strftime("%H:%M:%S"),

                "Prediction": prediction,

                "Confidence": f"{confidence:.2%}",

                "Exit": exit_used,

            }

        )

        self.logs = self.logs[:20]

        runtime = int(

            time.time()

            - self.start_time

        )

        self.header.metric(

            "Runtime",

            f"{runtime} sec",

        )

        self.show_prediction(

            prediction,

            confidence,

            exit_used,

            inference_time,

        )

        self.show_performance(

            packets,

            flows,

            average_packets,

            latency_saved,

        )

        self.show_distribution()

        self.show_exit_usage()

        self.show_logs()
        # ---------------------------------------------------------
    # Prediction Card
    # ---------------------------------------------------------

    def show_prediction(

        self,

        prediction,

        confidence,

        exit_used,

        inference_time,

    ):

        self.current_prediction.markdown(

        f"""
        ### Current Prediction

        **Traffic Type**

        ## {prediction}

        Confidence

        **{confidence:.2%}**

        Early Exit

        **{exit_used}**

        Inference Time

        **{inference_time:.3f} ms**
        """
                )
    
        # ---------------------------------------------------------
    # Traffic Distribution
    # ---------------------------------------------------------

    def show_distribution(self):

        if not self.class_counts:
            return

        df = pd.DataFrame({

            "Traffic": list(self.class_counts.keys()),

            "Flows": list(self.class_counts.values())

        })

        self.traffic_chart.bar_chart(

            df.set_index("Traffic")

        )

        total = sum(self.class_counts.values())

        table = []

        for cls, count in sorted(self.class_counts.items()):

            table.append({

                "Traffic": cls,

                "Flows": count,

                "Percentage": f"{100*count/total:.2f}%"

            })

        self.class_table.dataframe(

            pd.DataFrame(table),

            use_container_width=True,

            hide_index=True,

        )
        # ---------------------------------------------------------
    # Early Exit Usage
    # ---------------------------------------------------------

    def show_exit_usage(self):

        if not self.exit_counts:
            return

        total = sum(self.exit_counts.values())

        data = []

        for exit_name in [

            "exit1",

            "exit2",

            "exit3",

            "final",

        ]:

            count = self.exit_counts.get(

                exit_name,

                0,

            )

            data.append({

                "Exit": exit_name,

                "Count": count,

                "Percentage": f"{100*count/total:.2f}%"

            })

        self.exit_usage.markdown("### Early Exit Usage")

        self.exit_usage.dataframe(

            pd.DataFrame(data),

            hide_index=True,

            use_container_width=True,

        )
        # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    def show_performance(

        self,

        packets,

        flows,

        average_packets,

        latency_saved,

    ):

        self.performance.markdown(

        f"""
        ### Performance

        Packets Captured

        ## {packets}

        Flows Classified

        ## {flows}

        Average Packets

        **{average_packets:.2f}/20**

        Computation Saved

        **{latency_saved:.2f}%**
        """
                )
        # ---------------------------------------------------------
    # Event Log
    # ---------------------------------------------------------

    def show_logs(self):

        self.event_log.markdown(

            "### Recent Predictions"

        )

        self.event_log.dataframe(

            pd.DataFrame(self.logs),

            hide_index=True,

            use_container_width=True,

        )