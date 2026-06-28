from typing import Dict, List, Optional


class TemporalNormalizer:
    """
    Novelty-1 Temporal Normalization

    Purpose
    -------
    Reduce network-specific timing effects by scaling
    packet timing features using handshake-derived latency.

    Modes
    -----
    strict:
        TCP -> TLS -> NONE

    fallback:
        TCP -> TLS -> PROXY

    Backward Integration
    --------------------
    - handshake_detector.py

    Forward Integration
    -------------------
    - feature_encoder.py
    """

    MIN_BASELINE = 1e-6

    def __init__(
        self,
        mode="strict",
        feature_encoder=None
    ):

        self.mode = mode

        self.feature_encoder = (
            feature_encoder
        )

    # =====================================================
    # Public API
    # =====================================================

    def process_sequence(
        self,
        sequence_record
    ) -> Dict:
        """
        Parameters
        ----------
        sequence_record : SequenceRecord

        """

        if sequence_record is None:
            return {}

        baseline_info = (
            self._select_baseline(sequence_record)
        )

        raw_sequence = (
            self._build_raw_sequence(
                sequence_record.sequence
            )
        )

        normalized_sequence = raw_sequence

        normalization_applied = False

        if baseline_info[
            "baseline_available"
        ]:

            normalized_sequence = (
                self._normalize_sequence(
                    raw_sequence,
                    baseline_info[
                        "baseline_latency"
                    ]
                )
            )

            normalization_applied = True

        result = {

            "normalization_applied":
                normalization_applied,

            "normalization_mode":
                self.mode,

            "baseline_available":
                baseline_info[
                    "baseline_available"
                ],

            "baseline_type":
                baseline_info.get(
                    "baseline_type"
                ),

            "baseline_latency":
                baseline_info.get(
                    "baseline_latency"
                ),

            "raw_sequence":
                raw_sequence,

            "normalized_sequence":
                normalized_sequence
        }

        self._attach_to_sequence(
            sequence_record,
            result
        )

        self._forward_integrations(
            sequence_record
        )

        return result

    # =====================================================
    # Baseline Selection
    # =====================================================

    def _select_baseline(
    self,
    sequence_record
) -> Dict:

        if  sequence_record is None:

            return {
                "baseline_available":
                    False
            }

        baseline_info = getattr(
    sequence_record,
    "baseline_info",
    {}
)
        if baseline_info.get(
            "baseline_available",
            False
        ):

            latency = baseline_info.get(
                "baseline_latency"
            )

            if self._is_valid_latency(
                sequence_record,
                latency
):

                return {
                    "baseline_available":
                        True,

                    "baseline_type":
                        baseline_info.get(
                            "baseline_type"
                        ),

                    "baseline_latency":
                        max(
                            float(latency),
                            self.MIN_BASELINE
                        )
                }

        if self.mode == "strict":

            return {
                "baseline_available":
                    False
            }

        proxy = self._generate_proxy_baseline(
    sequence_record
)

        if proxy[
            "baseline_available"
        ]:
            return proxy

        return {
            "baseline_available":
                False
        }

    # =====================================================
    # Proxy Baseline
    # =====================================================

    def _generate_proxy_baseline(
        self,
        sequence_record=None
    ) -> Dict:

        if  sequence_record is None:

            return {
                "baseline_available":
                    False
            }

        packets = sequence_record.sequence

        if len(packets) < 2:

            return {
                "baseline_available":
                    False
            }
        
        real_packets = [

    pkt

    for pkt in packets

    if not pkt.get(
        "is_padding",
        False
    )
]

        timestamps = []

        for pkt in real_packets[:10]:

            timestamps.append(
                float(
                    pkt["timestamp"]
                )
            )

        gaps = []

        for i in range(
            1,
            len(timestamps)
        ):

            gap = (
                timestamps[i]
                - timestamps[i - 1]
            )

            if gap > 0:
                gaps.append(gap)

        if not gaps:

            return {
                "baseline_available":
                    False
            }

        gaps.sort()

        n = len(gaps)

        if n % 2 == 1:

            median_gap = gaps[n // 2]

        else:

            median_gap = (
                gaps[n // 2 - 1]
                + gaps[n // 2]
            ) / 2

        median_gap = max(
            median_gap,
            self.MIN_BASELINE
        )

        return {

            "baseline_available":
                True,

            "baseline_type":
                "proxy",

            "baseline_latency":
                median_gap
        }

    # =====================================================
    # Raw Timing Feature Creation
    # =====================================================

    def _build_raw_sequence(
    self,
    packets
) -> List[Dict]:

        if not packets:
            return []

        first_timestamp = None
        previous_timestamp = None

        sequence = []

        for pkt in packets:

            packet = dict(pkt)

            if packet.get(
                "is_padding",
                False
            ):

                packet["relative_timestamp"] = 0.0
                packet["inter_arrival_time"] = 0.0

                sequence.append(packet)

                continue

            timestamp = float(
                packet["timestamp"]
            )

            if first_timestamp is None:
                first_timestamp = timestamp

            relative_timestamp = (
                timestamp
                - first_timestamp
            )

            if previous_timestamp is None:

                iat = 0.0

            else:

                iat = (
                    timestamp
                    - previous_timestamp
                )

            packet["relative_timestamp"] = (
                relative_timestamp
            )

            packet["inter_arrival_time"] = (
                iat
            )

            sequence.append(packet)

            previous_timestamp = timestamp

        return sequence

    # =====================================================
    # Normalization
    # =====================================================

    def _normalize_sequence(
    self,
    sequence,
    baseline
):

        normalized = []

        for packet in sequence:

            item = dict(packet)

            if item.get(
                "is_padding",
                False
            ):

                normalized.append(item)

                continue

            item["relative_timestamp"] = (
                item["relative_timestamp"]
                / baseline
            )

            item["inter_arrival_time"] = (
                item["inter_arrival_time"]
                / baseline
            )

            # Use normalized relative time as timestamp
            item["timestamp"] = (
                item["relative_timestamp"]
            )

            normalized.append(item)

        return normalized

    # =====================================================
    # Validation
    # =====================================================

    def _is_valid_latency(
        self,
        sequence_record,
        latency
    ):

        if latency is None:
            return False

        latency = float(latency)

        if latency <= 0:
            return False

        real_packets = [

            pkt

            for pkt in sequence_record.sequence

            if not pkt.get(
                "is_padding",
                False
            )
        ]
        if len(real_packets) < 2:
            return False
        flow_duration = (

        float(
            real_packets[-1][
                "timestamp"
            ]
        )

        -

        float(
            real_packets[0][
                "timestamp"
            ]
        )
    )

        if latency > flow_duration:
            return False

        return True

    # =====================================================
    # Forward Integration
    # =====================================================

    def _forward_integrations(
        self,
        sequence_record
    ):

        if (
            self.feature_encoder
            and hasattr(
                self.feature_encoder,
                "receive_normalized_sequence"
            )
        ):

            self.feature_encoder.\
                receive_normalized_sequence(
                    sequence_record
                )

    # =====================================================
    # Attachment
    # =====================================================

    def _attach_to_sequence(
        self,
        sequence_record,
        result
    ):

        sequence_record.temporal_normalized = (
            result[
                "normalization_applied"
            ]
        )

        sequence_record.normalization_mode = (
            result[
                "normalization_mode"
            ]
        )

        sequence_record.normalization_baseline_type = (
            result[
                "baseline_type"
            ]
        )

        sequence_record.normalization_baseline_latency = (
            result[
                "baseline_latency"
            ]
        )

        sequence_record.normalization_baseline_available = (
            result[
                "baseline_available"
            ]
        )

        sequence_record.raw_sequence = (
            result[
                "raw_sequence"
            ]
        )

        sequence_record.normalized_sequence = (
            result[
                "normalized_sequence"
            ]
        )