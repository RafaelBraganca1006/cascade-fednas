from abc import ABC, abstractmethod
import wandb
import os
import logging
import torch
import numpy as np
from wandb.sdk.lib import RunDisabled
import copy
import random
import heapq


class BaseServerModel(ABC):
    def __init__(
        self, init_params, sampling_method, num_cli_total, cli_subnet_track=None,
    ):
        self.model = self.init_model(init_params)
        self.sampling_method = sampling_method
        if cli_subnet_track is None:
            self.cli_subnet_track = dict()
            for idx in range(num_cli_total):
                self.cli_subnet_track[idx] = dict()
                self.cli_subnet_track[idx]["largest"] = 0
                self.cli_subnet_track[idx]["smallest"] = 0
        else:
            self.cli_subnet_track = cli_subnet_track
        self.client_sample_count = None
        self.cli_indices = []
        self.top_k = 1
        self.bottom_k = 1
        self.largest_subnet_min_idx = set()
        self.smallest_subnet_min_idx = set()
        self.max_sample_count_idx = -1
        self.second_max_sample_count_idx = -1
        self.cur_round = -1
        self.cur_arch = None
        self.subnet_sampling = dict()
        self.subnet_sampling["static"] = self.static_sample
        self.subnet_sampling["dynamic"] = self.dynamic_sample
        self.subnet_sampling["all_random"] = self.random_subnet_sample
        self.subnet_sampling["compound"] = self.compound_subnet_sample
        self.subnet_sampling["sandwich_all_random"] = self.sandwich_all_subnet_sample
        self.subnet_sampling["sandwich_compound"] = self.sandwich_compound_subnet_sample
        self.subnet_sampling["TS_all_random"] = self.tracking_sandwich_all_subnet_sample
        self.subnet_sampling[
            "TS_compound"
        ] = self.tracking_sandwich_compound_subnet_sample
        self.subnet_sampling["max_sample_count"] = self.max_client_dataset_all_subnet
        self.subnet_sampling["multi_sandwich"] = self.multi_sandwich_sample
        self.subnet_sampling["TS_KD"] = self.tracking_sandwich_kd
        self.subnet_sampling["PS"] = self.ps
        self.subnet_sampling["capacity_aware"] = self.capacity_aware_sample
        self.subnet_sampling["capacity_rank"] = self.capacity_rank_sample
        self.subnet_sampling["policy_assign"] = self.policy_assign_sample

        # dynFedNAS capacity-aware sampling state (Phase 3). cap_bw is None when
        # comm_sim is off, in which case capacity_aware degenerates to the
        # TS_all_random behaviour (bandwidth never restricts anything).
        self.cap_bw = None
        self.cap_deadline = None
        self.cap_bytes_per_param = 4
        self.cap_num_clamped = 0
        self.cap_max_infeasible = 0
        self._cap_tier_bits = None
        self._cap_max_bits = None

        # assignment-policy sampling state ("policy_assign", DIRECTION §3.2,
        # retargeted 2026-07-22: "who trains what" under nested transmission).
        # Reuses cap_bw/cap_deadline above for bandwidth context (set via the
        # same set_capacity_info() the capacity_aware sampler uses) -- only
        # _policy and the id-carrying tier menu are new.
        self._policy = None
        self._policy_tier_options = None

    def set_top_bottom_k(self, top_k, bottom_k):
        self.top_k = top_k
        self.bottom_k = bottom_k

    def load_client_sample_counts(self, sample_counts):
        self.client_sample_count = sample_counts

    def update_sample(self):
        self.largest_subnet_min_idx = set()
        self.smallest_subnet_min_idx = set()
        temp = set()
        heap = []
        available_cli = set()
        for cli in self.cli_indices:
            available_cli.add(cli)
        # capacity-aware: the largest-subnet slot may only go to clients whose
        # current bandwidth fits the max subnet within the deadline, so the
        # scheduler's maxnet aggregation weight always lands on a real max update
        largest_pool = available_cli
        if self.cap_bw is not None and self._cap_max_bits is not None:
            eligible = set(
                cli
                for cli in available_cli
                if self.cap_bw.get(cli, 0.0) * self.cap_deadline
                >= self._cap_max_bits
            )
            if len(eligible) > 0:
                largest_pool = eligible
            else:
                # no client can afford the max subnet this round: best-effort
                # attempt by the fastest client (same failure mode as vanilla,
                # never worse); aggregation-side robustness is Phase 4
                self.cap_max_infeasible += 1
                largest_pool = {
                    max(available_cli, key=lambda cli: self.cap_bw.get(cli, 0.0))
                }
        for i in largest_pool:
            heapq.heappush(heap, self.cli_subnet_track[i]["largest"])
        for i in range(min(self.top_k, len(heap))):
            temp.add(heap[i])
        for cli in largest_pool:
            if len(self.largest_subnet_min_idx) >= self.top_k:
                break
            if self.cli_subnet_track[cli]["largest"] in temp:
                self.largest_subnet_min_idx.add(cli)
        available_cli = available_cli - self.largest_subnet_min_idx
        # of the remaining clients temporally load balance min subnet
        heap = []
        for i in available_cli:
            heapq.heappush(heap, self.cli_subnet_track[i]["smallest"])
        temp = set()
        for i in range(self.bottom_k):
            temp.add(heap[i])
        for cli in available_cli:
            if len(self.smallest_subnet_min_idx) >= self.bottom_k:
                break
            if self.cli_subnet_track[cli]["smallest"] in temp:
                self.smallest_subnet_min_idx.add(cli)

        max_sample_count_idx = -1
        for idx in range(len(self.cli_indices)):
            cur_idx = self.cli_indices[idx]
            if (
                    max_sample_count_idx == -1
                    or self.client_sample_count[cur_idx]
                    > self.client_sample_count[max_sample_count_idx]
            ):
                max_sample_count_idx = cur_idx
        # Find min smallest
        second_max_sample_count_idx = -1
        for idx in range(len(self.cli_indices)):
            cur_idx = self.cli_indices[idx]
            if cur_idx != max_sample_count_idx:
                if (
                    second_max_sample_count_idx == -1
                    or self.client_sample_count[cur_idx]
                    > self.client_sample_count[second_max_sample_count_idx]
                ):
                    second_max_sample_count_idx = cur_idx
        self.max_sample_count_idx = max_sample_count_idx
        self.second_max_sample_count_idx = second_max_sample_count_idx

    def set_cli_indices(self, client_indices):
        self.cli_indices = client_indices

    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_model_copy(self):
        return copy.deepcopy(self.model)

    # superimpose a (numpy) vectorized version of it's MAX network onto supernetwork
    def superimpose_vec(self, vec):
        with torch.no_grad():
            vec = torch.from_numpy(vec)
            idx = 0
            for p in self.model.parameters():
                idx_next = idx + p.view(-1).size()[0]
                p.copy_(vec[idx:idx_next].reshape(p.shape))
                idx = idx_next

    # sums supernetwork with a (numpy) vectorized version of it's MAX network in place
    def sum_supernet_w_vec(self, vec):
        with torch.no_grad():
            vec = torch.from_numpy(vec)
            idx = 0
            for p in self.model.parameters():
                idx_next = idx + p.view(-1).size()[0]
                p.copy_(p.data + vec[idx:idx_next].reshape(p.shape))
                idx = idx_next

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def save(self, name, supernet_config, supernet_class):
        if isinstance(wandb.run, RunDisabled):
            return
        save_data = dict()
        save_data["params"] = self.get_model_params()
        save_data["supernet_config"] = supernet_config
        save_data["supernet_class"] = supernet_class
        save_data["torch_rng_state"] = torch.get_rng_state()
        save_data["numpy_rng_state"] = np.random.get_state()
        save_data["cli_subnet_track"] = self.cli_subnet_track
        filename = os.path.join(wandb.run.dir, name)
        torch.save(
            save_data, filename,
        )
        wandb.save(filename)

    #TODO: Abstract this function away and allow different supernetworks to specify expected input size
    def wandb_pass(self):
        self.model.set_max_net()
        self.model.eval()
        self.model(torch.zeros((1, 3, 32, 32)))

    def feddyn_global_model_update(self, alpha):
        for server_param, state_param in zip(
            self.model.parameters(), self.server_state.parameters()
        ):
            server_param.data -= (1 / alpha) * state_param

    def sample_subnet(self, round_num, client_idx, idx, args):
        return self.subnet_sampling[self.sampling_method](
            round_num, client_idx, idx, args
        )

    def static_sample(self, round_num, client_idx, idx, args):
        subnets = args["diverse_subnets"]
        id = str(client_idx % len(subnets))
        return self.get_subnet(**subnets[id])

    def dynamic_sample(self, round_num, client_idx, idx, args):
        subnets = args["diverse_subnets"]
        id = str((client_idx + round_num) % len(subnets))
        return self.get_subnet(**subnets[id])

    def random_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        arch_config = self.random_subnet_arch()
        return self.get_subnet(**arch_config)

    def compound_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        arch_config = self.random_compound_subnet_arch()
        return self.get_subnet(**arch_config)

    def sandwich_all_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if idx == (round_num % args["client_per_round"]):
            arch_config = self.min_subnet_arch()
        elif idx == ((round_num + 1) % args["client_per_round"]):
            arch_config = self.max_subnet_arch()
        else:
            arch_config = self.random_subnet_arch()
        return self.get_subnet(**arch_config)

    def sandwich_compound_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if idx == (round_num % args["client_per_round"]):
            arch_config = self.min_subnet_arch()
        elif idx == ((round_num + 1) % args["client_per_round"]):
            arch_config = self.max_subnet_arch()
        else:
            arch_config = self.random_compound_subnet_arch()
        return self.get_subnet(**arch_config)

    def tracking_sandwich_all_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def tracking_sandwich_compound_subnet_sample(
        self, round_num, client_idx, idx, args
    ):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_compound_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def max_client_dataset_all_subnet(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if client_idx == self.second_max_sample_count_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx == self.max_sample_count_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)
    def multi_sandwich_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        model_dict = dict()
        model_dict["max"] = self.get_subnet(**self.max_subnet_arch())
        model_dict["min"] = self.get_subnet(**self.min_subnet_arch())
        model_dict["mid"] = []
        for _ in range(args["K"]):
            model_dict["mid"].append(self.get_subnet(**self.random_subnet_arch()))
        return model_dict

    def tracking_sandwich_kd(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if client_idx == self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            model_dict = dict()
            model_dict["max"] = None
            model_dict["min"] = None
            model_dict["mid"] = [self.get_subnet(**self.max_subnet_arch())]
            self.cli_subnet_track[client_idx]["largest"] += 1
            return model_dict
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        model_dict = dict()
        model_dict["max"] = self.get_subnet(**self.max_subnet_arch())
        model_dict["min"] = None
        model_dict["mid"] = [arch_config]
        return model_dict

    def ps(self, round_num, client_idx, idx, args):
        if self.cur_round < round_num:
            self.cur_round = round_num
            #np.random.seed(round_num)
            if self.cur_round < args["ps_depth_only"]:
                self.cur_arch = self.random_depth_subnet_arch()
            else:
                self.cur_arch = self.random_subnet_arch()
        return self.get_subnet(**self.cur_arch)

    # ---- dynFedNAS: capacity-aware sampling (Phase 3) ----
    def set_capacity_info(
        self, bw_bits_per_sec, deadline_s, sampler_args, bytes_per_param=4
    ):
        """Per-round bandwidth picture for the capacity_aware sampler.

        bw_bits_per_sec: dict client_idx -> estimated uplink (bits/sec), or None
        to disable. Must be called before update_sample() each round so the
        largest-subnet slot (and with it the maxnet aggregation weight) is only
        drawn from clients that can afford the max subnet's payload.
        """
        self.cap_bw = bw_bits_per_sec
        self.cap_deadline = deadline_s
        self.cap_bytes_per_param = bytes_per_param
        self.cap_num_clamped = 0
        self.cap_max_infeasible = 0
        if bw_bits_per_sec is not None:
            self._ensure_capacity_payloads(sampler_args)

    def _payload_bits(self, client_model):
        # same sizing rule as CommSimulator.payload_bits, so the size used to
        # assign matches the size charged at transmit time
        num_params = sum(
            p.numel() for p in client_model.model.parameters() if p.requires_grad
        )
        return num_params * self.cap_bytes_per_param * 8

    def _ensure_capacity_payloads(self, args):
        if self._cap_tier_bits is not None:
            return
        tiers = []
        for subnet_id in args["diverse_subnets"]:
            config = args["diverse_subnets"][subnet_id]
            tiers.append((self._payload_bits(self.get_subnet(**config)), config))
        tiers.sort(key=lambda t: t[0])
        self._cap_tier_bits = tiers
        self._cap_max_bits = self._payload_bits(
            self.get_subnet(**self.max_subnet_arch())
        )
        logging.info(
            "[capacity] tier payloads: %s | max subnet: %.1f MB"
            % (
                ", ".join(
                    "%s -> %.1f MB" % (str(cfg), bits / 8e6)
                    for bits, cfg in tiers
                ),
                self._cap_max_bits / 8e6,
            )
        )

    def _budget_bits(self, client_idx):
        return self.cap_bw.get(client_idx, 0.0) * self.cap_deadline

    def _largest_affordable_tier(self, budget_bits):
        # smallest configured tier if nothing fits (best-effort contribution)
        config = self._cap_tier_bits[0][1]
        for bits, tier_config in self._cap_tier_bits:
            if bits <= budget_bits:
                config = tier_config
        return config

    def capacity_aware_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
            return self.get_subnet(**arch_config)
        if client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
            return self.get_subnet(**arch_config)
        # middle clients: same random draw (and seed) as TS_all_random; bandwidth
        # only acts as a ceiling — degrade to the largest tier that fits, so the
        # sampling distribution matches the baseline whenever capacity suffices
        arch_config = self.random_subnet_arch()
        client_model = self.get_subnet(**arch_config)
        if self.cap_bw is not None:
            budget_bits = self._budget_bits(client_idx)
            if self._payload_bits(client_model) > budget_bits:
                arch_config = self._largest_affordable_tier(budget_bits)
                client_model = self.get_subnet(**arch_config)
                self.cap_num_clamped += 1
        if self.is_max_net(arch_config):
            self.cli_subnet_track[client_idx]["largest"] += 1
        elif self.is_min_net(arch_config):
            self.cli_subnet_track[client_idx]["smallest"] += 1
        return client_model

    def capacity_rank_sample(self, round_num, client_idx, idx, args):
        """MaxFit's successor: assign the ladder by BANDWIDTH RANK, not by a
        threshold on what fits.

        Why capacity_aware is not enough. It draws the architecture exactly like
        TS_all_random and then only ever CLAMPS DOWN when the payload exceeds the
        budget -- it never assigns up. So a well-connected client that happened to
        draw S keeps S, and the scarcest resource in the federation (a good link,
        the only kind that can carry the top of the ladder) is spent on the
        cheapest region. Under a tight deadline the clamp also shifts the whole
        architecture distribution downward, which is a plausible cause of the flat
        front capacity_aware produces (spread -0.31 at alpha=100/T=20).

        The rule here is relative rather than absolute: of the round's middle
        clients, the one with the best estimated budget gets the largest tier, the
        worst gets the smallest, monotonically in between. Nobody's good link is
        wasted, and the top of the ladder is assigned to SOMEBODY every round even
        when no client could deliver it in full.

        That last part is only safe because delivery is anytime (DIRECTION 3.1):
        under all-or-nothing, assigning a subnet the client cannot finish is a
        guaranteed loss and this rule would be strictly worse than the clamp;
        under nested/nested_blocks it is a partial contribution. The mechanism is
        what makes the more aggressive assignment affordable -- it does not
        replace the assignment layer, it changes what that layer can do.

        Deliberate consequence, to state rather than hide: this makes the
        architecture distribution bandwidth-CORRELATED, where TS_all_random's is
        uniform over the elastic space (which is OFA's supernet-training
        premise). That is the treatment, not a confound -- but it does mean a
        comparison against blind measures rule AND distribution together, exactly
        the caveat DIRECTION 3.2b fact 4 raised about policy_assign.
        """
        # The forced min/max slots are MaxNet's sandwich and are left untouched:
        # capacity_aware already restricts the max slot to clients that can
        # afford it, so the top of the ladder is not the problem being fixed.
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            self.cli_subnet_track[client_idx]["smallest"] += 1
            return self.get_subnet(**self.min_subnet_arch())
        if client_idx in self.largest_subnet_min_idx:
            self.cli_subnet_track[client_idx]["largest"] += 1
            return self.get_subnet(**self.max_subnet_arch())

        # No bandwidth picture (comm_sim off) -> fall back to the blind draw, so
        # this sampler degenerates to TS_all_random exactly like capacity_aware.
        if self.cap_bw is None or not self._cap_tier_bits:
            np.random.seed(round_num + client_idx)
            return self.get_subnet(**self.random_subnet_arch())

        middle = [
            c for c in self.cli_indices
            if c not in self.smallest_subnet_min_idx
            and c not in self.largest_subnet_min_idx
        ]
        # (bandwidth, client) so ties break deterministically
        order = sorted(middle, key=lambda c: (self.cap_bw.get(c, 0.0), c))
        try:
            rank = order.index(client_idx)
        except ValueError:                      # not a middle client this round
            np.random.seed(round_num + client_idx)
            return self.get_subnet(**self.random_subnet_arch())

        n_tiers = len(self._cap_tier_bits)
        tier = (rank * n_tiers) // max(len(order), 1)
        arch_config = self._cap_tier_bits[min(tier, n_tiers - 1)][1]
        client_model = self.get_subnet(**arch_config)
        if self.is_max_net(arch_config):
            self.cli_subnet_track[client_idx]["largest"] += 1
        elif self.is_min_net(arch_config):
            self.cli_subnet_track[client_idx]["smallest"] += 1
        return client_model

    def set_assignment_policy(self, policy):
        """Register the AssignmentPolicy object for the "policy_assign"
        sampler (DIRECTION §3.2). Call once at trainer init -- the policy
        instance persists across rounds (stateful policies like Lyapunov/
        the bandit carry their own queues/weights internally)."""
        self._policy = policy

    def _ensure_policy_tier_options(self, args):
        if self._policy_tier_options is not None:
            return self._policy_tier_options
        from fedml_api.standalone.superfednas.comm_sim.policies import TierOption

        options = []
        for tier_id, config in args["diverse_subnets"].items():
            bits = self._payload_bits(self.get_subnet(**config))
            options.append(TierOption(tier_id=tier_id, config=config, payload_bits=bits))
        options.sort(key=lambda t: t.payload_bits)
        self._policy_tier_options = options
        return options

    def policy_assign_sample(self, round_num, client_idx, idx, args):
        """--subnet_dist_type policy_assign: delegates tier choice to the
        registered AssignmentPolicy (greedy/lyapunov/bandit, DIRECTION §3.2)
        for "middle" clients, same min/max guaranteed-coverage slots as
        every other TS-style sampler (keeps architecture-space coverage
        intact regardless of which policy is active)."""
        from fedml_api.standalone.superfednas.comm_sim.policies import AssignmentContext

        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
            return self.get_subnet(**arch_config)
        if client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
            return self.get_subnet(**arch_config)
        if self._policy is None:
            raise RuntimeError(
                "policy_assign sampler active but no policy registered -- "
                "call set_assignment_policy() before update_sample()."
            )
        tier_options = self._ensure_policy_tier_options(args)
        bw_estimate = self.cap_bw.get(client_idx) if self.cap_bw is not None else None
        tier_debt = args.get("tier_debt")
        debt_by_tier = (
            {tid: round_num - last for tid, last in tier_debt.items()}
            if tier_debt is not None
            else None
        )
        client_tier_debt_table = args.get("client_tier_debt")
        client_tier_debt = (
            {
                tid: round_num - last
                for tid, last in client_tier_debt_table.get(client_idx, {}).items()
            }
            if client_tier_debt_table is not None
            else None
        )
        data_utility = args.get("data_utility", {}).get(client_idx)
        ctx = AssignmentContext(
            round_num=round_num,
            client_idx=client_idx,
            tier_options=tier_options,
            bw_estimate_bps=bw_estimate,
            upload_window_s=self.cap_deadline,
            debt_by_tier=debt_by_tier,
            client_tier_debt=client_tier_debt,
            data_utility=data_utility,
        )
        chosen = self._policy.assign(ctx)
        arch_config = chosen.config
        if self.is_max_net(arch_config):
            self.cli_subnet_track[client_idx]["largest"] += 1
        elif self.is_min_net(arch_config):
            self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def random_subnet_arch_under(self, cap_config):
        """A random elastic arch, clipped by a size CAP (E6b, DIRECTION §3.2b).

        This is the action the retargeted policy takes: not "train tier k", but
        "train anything up to size k". The clip is the same alpha ^ tier
        intersection the transmission layer already uses (comm_sim.transmission.
        clip_arch_config, min per stage/block), so the result is a valid OFA
        config nested inside the cap.

        THE property this exists for: at cap = the max tier the clip is the
        IDENTITY (every d <= 2, every e <= 0.25), and the numpy draw sequence is
        untouched, so `cap = XL` reproduces TS_all_random BIT-IDENTICALLY. Blind
        sampling is therefore a member of the policy class, which is what makes
        a loss attributable to the policy rather than to a restricted action
        space -- the confound that invalidated the E5/E6 blind comparison
        (DIRECTION §3.2b fact 4).

        Note what this does NOT give: reachability is not a performance floor.
        The policy optimizes its reward, not accuracy, and will leave cap = XL
        confidently if the reward is misaligned -- exactly what the v1 bandit
        did. Asserted by selftest_selection.py::test_cap_max_equals_blind.
        """
        from fedml_api.standalone.superfednas.comm_sim.transmission import (
            clip_arch_config,
        )

        arch_config = self.random_subnet_arch()
        if cap_config is None:
            return arch_config
        return clip_arch_config(arch_config, cap_config)

    def crossover_sample(self, par1, par2):
        new_sample = copy.deepcopy(par1)
        for key in new_sample.keys():
            if not isinstance(new_sample[key], list):
                continue
            for i in range(len(new_sample[key])):
                new_sample[key][i] = random.choice([par1[key][i], par2[key][i]])
        return new_sample

    # TODO:Need to reimpliment updated sandwich sampling with dynamic width
    def dynamic_width_sandwich_sample(self, round_num, client_idx, idx, args):
        pass

    @abstractmethod
    def init_model(self, init_params):
        pass

    @abstractmethod
    def is_max_net(self, arch):
        pass

    @abstractmethod
    def is_min_net(self, arch):
        pass

    @abstractmethod
    def get_subnet(self, **kwargs):
        pass

    @abstractmethod
    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        pass

    @abstractmethod
    def active_subnet_index(self):
        pass

    @abstractmethod
    def max_subnet_arch(self):
        pass

    @abstractmethod
    def min_subnet_arch(self):
        pass

    @abstractmethod
    def random_subnet_arch(self):
        pass

    @abstractmethod
    def random_depth_subnet_arch(self):
        pass

    @abstractmethod
    def random_compound_subnet_arch(self):
        pass

    @abstractmethod
    def mutate_sample(self, sample_arch, mut_prob):
        pass
