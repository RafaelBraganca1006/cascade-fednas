"""Communication simulator for dynFedNAS (v2).

Sits on top of the SuperFedNAS training loop. For each participating client in
a round it:

  1. sizes the payload of the assigned subnet (params x bytes x 8 = bits),
  2. reads the client's current uplink from a ``BandwidthModel``,
  3. computes the client's upload budget = bandwidth x (deadline - t_compute),
  4. decides what arrives according to the transmission mode:
       all_or_nothing  the whole update fits or it is dropped (Phase 1-3),
       nested          the deadline truncates a shell-ordered stream; every
                       complete prefix is a valid smaller-tier update,
       nested_shard    nested + top-shell sharding (Stage 3).

Time model (DIRECTION.md §6): the deadline is the round's time budget; local
compute is the deterministic share (``t_compute``, 0 in the main experiments —
homogeneous compute), communication the stochastic share under study. A
synchronous server cannot know a straggler will miss the deadline before the
deadline: any round with an incomplete upload lasts the full deadline.

Metrics: participation, per-client drops, max-subnet failure, attempted vs
delivered vs wire-wasted bits (the by-construction zero-waste claim needs the
distinction), and communication wall-clock.
"""

from collections import defaultdict

from .transmission import TX_MODES, all_or_nothing


class CommSimulator:
    def __init__(self, bandwidth_model, deadline_s, bytes_per_param=4,
                 tx_mode="all_or_nothing"):
        assert tx_mode in TX_MODES, tx_mode
        self.bw = bandwidth_model
        self.deadline = float(deadline_s)
        self.bytes_per_param = bytes_per_param
        self.tx_mode = tx_mode

        # cumulative, whole-run
        self.wall_clock = 0.0
        self.per_client_selected = defaultdict(int)
        self.per_client_dropped = defaultdict(int)
        self.total_bits_sent = 0.0        # delivered (usable) bits
        self.total_bits_dropped = 0.0     # payload bits of voided updates (legacy metric)
        self.total_bits_attempted = 0.0   # bits pushed onto the wire
        self.total_bits_wasted = 0.0      # attempted but unusable (wire waste)
        self.round_records = []

        self._cur = None  # per-round accumulator

    # ----- round time budget -----
    def t_compute(self, client_idx, config=None):
        """Deterministic local-compute share of the round budget, seconds.

        0 in the main experiments (homogeneous compute — see DIRECTION.md §6:
        a known constant only shifts the window). The heterogeneous-window
        ablation (T_i = T - c_i) and a future three-phase model plug in here.
        """
        return 0.0

    def upload_window(self, client_idx, config=None):
        """Seconds of the deadline left for uploading."""
        return max(self.deadline - self.t_compute(client_idx, config), 0.0)

    def upload_budget_bits(self, client_idx, round_num, config=None):
        bw = max(self.bw.bandwidth(client_idx, round_num), 1e-9)  # bits/sec
        return bw * self.upload_window(client_idx, config), bw

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
            "bits_attempted": 0.0,
            "bits_wasted": 0.0,
            "max_success_upload_time": 0.0,
            "incomplete": 0,  # uploads the deadline cut short (drop or truncation)
            "delivered_tiers": [],  # per transmit_plan client: delivered tier index
            "units_delivered": 0,  # shard units delivered by carriers this round
        }

    def _record(self, client_idx, is_max_net, payload_bits, report):
        """Fold one client's DeliveryReport into the round accumulators."""
        c = self._cur
        c["selected"] += 1
        self.per_client_selected[client_idx] += 1
        if is_max_net:
            c["max_present"] = True

        c["bits_attempted"] += report.bits_attempted
        c["bits_wasted"] += report.bits_wasted
        if not report.completed_plan:
            c["incomplete"] += 1

        if report.delivers_nothing():
            c["dropped"] += 1
            c["bits_dropped"] += payload_bits
            self.per_client_dropped[client_idx] += 1
            if is_max_net:
                c["max_failed"] = True
        else:
            c["success"] += 1
            c["bits_sent"] += report.bits_delivered
            c["max_success_upload_time"] = max(
                c["max_success_upload_time"], report.time_used
            )
            if is_max_net and not report.completed_plan:
                # the max holder delivered only a smaller tier
                c["max_failed"] = True

    def transmit(self, round_num, client_idx, client_model, is_max_net):
        """All-or-nothing transmission (legacy Phase 1-3 path).

        Returns (success: bool, upload_time: float). On failure the caller
        should drop the update (skip aggregation) for this client.
        """
        bits = self.payload_bits(client_model)
        budget, bw = self.upload_budget_bits(client_idx, round_num)
        report = all_or_nothing(
            bits, budget, bw, self.upload_window(client_idx)
        )
        self._record(client_idx, is_max_net, bits, report)
        return not report.delivers_nothing(), report.time_used

    def transmit_plan(self, round_num, client_idx, plan, is_max_net):
        """Nested-shell transmission (v2 mechanisms): deadline truncates the
        planned stream; returns a DeliveryReport (see transmission.py)."""
        from .transmission import truncate

        budget, bw = self.upload_budget_bits(client_idx, round_num)
        report = truncate(plan, budget, bw, self.upload_window(client_idx))
        self._record(client_idx, is_max_net, plan.total_bits(), report)
        self._cur["delivered_tiers"].append(report.delivered_tier)
        return report

    def transmit_shards(self, round_num, client_idx, shard_plan, is_max_net=False):
        """Carrier upload (nested_shard mode): whole shard units until the
        budget runs out. Returns (DeliveryReport, delivered unit ids)."""
        from .transmission import truncate_units

        budget, bw = self.upload_budget_bits(client_idx, round_num)
        report, delivered = truncate_units(
            shard_plan, budget, bw, self.upload_window(client_idx)
        )
        self._record(client_idx, is_max_net, shard_plan.total_bits(), report)
        self._cur["units_delivered"] += len(delivered)
        return report, delivered

    def end_round(self):
        """Finalize the round, advance wall-clock, return a summary dict."""
        c = self._cur
        # Synchronous round: the server cannot know an upload will miss the
        # deadline before the deadline, so any round with an incomplete upload
        # lasts the full deadline. Only when every planned upload completed
        # early does the round end at the slowest completion.
        if c["incomplete"] > 0 or c["success"] == 0:
            round_time = self.deadline
        else:
            round_time = c["max_success_upload_time"]
        self.wall_clock += round_time
        self.total_bits_sent += c["bits_sent"]
        self.total_bits_dropped += c["bits_dropped"]
        self.total_bits_attempted += c["bits_attempted"]
        self.total_bits_wasted += c["bits_wasted"]

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
            "comm/MB_attempted_round": c["bits_attempted"] / 8e6,
            "comm/MB_wasted_round": c["bits_wasted"] / 8e6,
            "comm/MB_attempted_total": self.total_bits_attempted / 8e6,
            "comm/MB_wasted_total": self.total_bits_wasted / 8e6,
            "comm/waste_rate": c["bits_wasted"] / max(c["bits_attempted"], 1e-9),
            "comm/full_plan_rate": (selected - c["incomplete"]) / selected,
        }
        if c["delivered_tiers"]:
            summary["comm/mean_delivered_tier"] = sum(c["delivered_tiers"]) / len(
                c["delivered_tiers"]
            )
        if c["units_delivered"]:
            summary["comm/shard_units_delivered"] = c["units_delivered"]
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
            "comm/total_MB_attempted": self.total_bits_attempted / 8e6,
            "comm/total_MB_wasted": self.total_bits_wasted / 8e6,
            "comm/overall_waste_rate": self.total_bits_wasted
            / max(self.total_bits_attempted, 1e-9),
            "comm/per_client_dropped": dict(self.per_client_dropped),
            "comm/per_client_selected": dict(self.per_client_selected),
        }
