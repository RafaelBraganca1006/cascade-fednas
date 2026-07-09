"""Communication simulator for dynFedNAS (Phase 1).

Sits on top of the (unmodified) SuperFedNAS training loop as a measurement
harness. For each participating client in a round it:

  1. sizes the payload of the assigned subnet (params x bytes x 8 = bits),
  2. reads the client's current uplink from a ``BandwidthModel``,
  3. computes upload_time = payload_bits / bandwidth,
  4. decides success = (upload_time <= deadline); a failed upload is *dropped*
     (excluded from aggregation), reproducing a real round-deadline timeout.

It accumulates the metrics the thesis reports: participation rate, per-client
drop counts, whether the critical largest-subnet update failed, communication
consumed, and wall-clock time. Vanilla SuperFedNAS assigns subnet sizes without
regard to bandwidth, so it drops updates here on its own — that degradation is
exactly what a bandwidth-aware method is meant to avoid.
"""

from collections import defaultdict


class CommSimulator:
    def __init__(self, bandwidth_model, deadline_s, bytes_per_param=4):
        self.bw = bandwidth_model
        self.deadline = float(deadline_s)
        self.bytes_per_param = bytes_per_param

        # cumulative, whole-run
        self.wall_clock = 0.0
        self.per_client_selected = defaultdict(int)
        self.per_client_dropped = defaultdict(int)
        self.total_bits_sent = 0.0
        self.total_bits_dropped = 0.0
        self.round_records = []

        self._cur = None  # per-round accumulator

    # ----- payload sizing -----
    def payload_bits(self, client_model):
        num_params = sum(
            p.numel() for p in client_model.model.parameters() if p.requires_grad
        )
        return num_params * self.bytes_per_param * 8

    # ----- round lifecycle -----
    def start_round(self, round_num):
        self._cur = {
            "round": round_num,
            "selected": 0,
            "success": 0,
            "dropped": 0,
            "max_present": False,
            "max_failed": False,
            "bits_sent": 0.0,
            "bits_dropped": 0.0,
            "max_success_upload_time": 0.0,
        }

    def transmit(self, round_num, client_idx, client_model, is_max_net):
        """Decide whether this client's update arrives in time.

        Returns (success: bool, upload_time: float). On failure the caller
        should drop the update (skip aggregation) for this client.
        """
        bits = self.payload_bits(client_model)
        bw = max(self.bw.bandwidth(client_idx, round_num), 1e-9)  # bits/sec
        upload_time = bits / bw
        success = upload_time <= self.deadline

        c = self._cur
        c["selected"] += 1
        self.per_client_selected[client_idx] += 1
        if is_max_net:
            c["max_present"] = True
        if success:
            c["success"] += 1
            c["bits_sent"] += bits
            c["max_success_upload_time"] = max(c["max_success_upload_time"], upload_time)
        else:
            c["dropped"] += 1
            c["bits_dropped"] += bits
            self.per_client_dropped[client_idx] += 1
            if is_max_net:
                c["max_failed"] = True
        return success, upload_time

    def end_round(self):
        """Finalize the round, advance wall-clock, return a summary dict."""
        c = self._cur
        # synchronous round: its duration is the slowest successful upload,
        # or the full deadline if nothing made it through (a wasted round).
        round_time = c["max_success_upload_time"] if c["success"] > 0 else self.deadline
        self.wall_clock += round_time
        self.total_bits_sent += c["bits_sent"]
        self.total_bits_dropped += c["bits_dropped"]

        selected = max(c["selected"], 1)
        summary = {
            "comm/participation_rate": c["success"] / selected,
            "comm/num_dropped": c["dropped"],
            "comm/num_selected": c["selected"],
            "comm/max_subnet_present": int(c["max_present"]),
            "comm/max_subnet_failed": int(c["max_failed"]),
            "comm/round_time_s": round_time,
            "comm/wall_clock_s": self.wall_clock,
            "comm/MB_sent_round": c["bits_sent"] / 8e6,
            "comm/MB_sent_total": self.total_bits_sent / 8e6,
        }
        self.round_records.append({**c, "round_time": round_time})
        self._cur = None
        return summary

    # ----- end-of-run reporting -----
    def summary(self):
        total_selected = sum(self.per_client_selected.values())
        total_dropped = sum(self.per_client_dropped.values())
        max_failed_rounds = sum(1 for r in self.round_records if r["max_failed"])
        max_present_rounds = sum(1 for r in self.round_records if r["max_present"])
        return {
            "comm/final_wall_clock_s": self.wall_clock,
            "comm/overall_drop_rate": total_dropped / max(total_selected, 1),
            "comm/max_subnet_failure_rate": max_failed_rounds / max(max_present_rounds, 1),
            "comm/total_MB_sent": self.total_bits_sent / 8e6,
            "comm/total_MB_dropped": self.total_bits_dropped / 8e6,
            "comm/per_client_dropped": dict(self.per_client_dropped),
            "comm/per_client_selected": dict(self.per_client_selected),
        }
