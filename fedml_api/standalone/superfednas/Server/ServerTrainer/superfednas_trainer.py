from abc import ABC, abstractmethod
import numpy as np
import logging
import copy
import wandb
import torch
from ofa.utils import flops_counter as fp


"""
Notes:
Local bn skipped
Don't support datasets with following format: self.args.dataset.startswith("stackoverflow"):
REMOVED CERTAIN SKIPS AND IF STATEMENTS IN ADD_SUBNET FUNCTIONS FOR AGGREGATION
PROXIMAL DIST LOSS NOT IMPLEMENTED
LAYERWISE WD NOT IMPLEMENTED
"""


# Note: note filtering non-trainable parameters
def model_vector(model, req_grad=True):
    if not req_grad:
        for p in model.parameters():
            p.requires_grad = False
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)


def model_dict_to_vector(model_dict, model_copy, req_grad=True):
    model_copy.load_state_dict(model_dict)
    return model_vector(model_copy, req_grad=req_grad)


class FLOFA_Trainer(ABC):
    def __init__(
        self,
        server_model,
        dataset,
        client_trainer,
        args,
        lr_scheduler,
        wt_avg_sched_method="Uniform",
        teacher_model=None,
        start_round=0,
    ):
        self.server_model = server_model
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.args = args
        self.start_round = start_round
        self.lr_scheduler = lr_scheduler
        self.teacher_model = teacher_model
        self.client_trainer = client_trainer
        self.sampler_args = dict()
        self.server_model.load_client_sample_counts(train_data_local_num_dict)
        self.server_model.set_top_bottom_k(self.args.top_k_maxnet, self.args.bottom_k_maxnet)
        self.sampler_args["client_per_round"] = self.args.client_num_per_round
        self.sampler_args["K"] = self.args.num_multi_archs
        self.sampler_args["diverse_subnets"] = self.args.diverse_subnets
        self.sampler_args["ps_depth_only"] = self.args.ps_depth_only

        # ---- dynFedNAS communication simulation (Phase 1 harness) ----
        self.comm_sim = None
        if getattr(self.args, "comm_sim", False):
            from fedml_api.standalone.superfednas.comm_sim import (
                CommSimulator,
                MarkovBandwidth,
                assign_connectivity_tiers,
            )

            if getattr(self.args, "bw_profile", "uniform").startswith("tiered"):
                # Heterogeneous connectivity (E4): persistent per-client
                # profiles, optionally correlated with the client's label
                # distribution (--bw_data_corr, digital-divide stress test).
                cls_counts = self._client_class_counts()
                # kept for the inclusion-gap metric (class-distribution-weighted
                # global accuracy per connectivity group -- see
                # _log_inclusion_gap); computed once here, never recomputed.
                self._cls_counts = cls_counts
                client_tiers = assign_connectivity_tiers(
                    cls_counts,
                    corr=getattr(self.args, "bw_data_corr", 0.0),
                    seed=self.args.bw_seed,
                )
                if self.args.bw_profile == "tiered_continuous":
                    # E6b (DIRECTION §3.2b): continuous within-state spread, so
                    # the whole tier ladder is reachable. Three discrete states
                    # give only three budgets at any deadline, which is what
                    # made the assignment problem degenerate.
                    bw_model = MarkovBandwidth.tiered_continuous(
                        num_clients=self.args.client_num_in_total,
                        num_rounds=self.args.comm_round,
                        client_tiers=client_tiers,
                        seed=self.args.bw_seed,
                        jitter_sigma=getattr(self.args, "bw_jitter_sigma", 0.45),
                    )
                else:
                    bw_model = MarkovBandwidth.tiered(
                        num_clients=self.args.client_num_in_total,
                        num_rounds=self.args.comm_round,
                        client_tiers=client_tiers,
                        seed=self.args.bw_seed,
                    )
                by_tier = {
                    name: [c for c, t in enumerate(client_tiers) if t == i]
                    for i, name in enumerate(MarkovBandwidth.TIER_NAMES)
                }
                logging.info(
                    "[comm_sim] connectivity tiers (corr=%.2f): %s"
                    % (getattr(self.args, "bw_data_corr", 0.0), by_tier)
                )
                logging.info(
                    "[comm_sim] client dominant classes: %s"
                    % {c: int(np.argmax(cls_counts[c])) for c in sorted(cls_counts)}
                )
            else:
                bw_model = MarkovBandwidth.default(
                    num_clients=self.args.client_num_in_total,
                    num_rounds=self.args.comm_round,
                    seed=self.args.bw_seed,
                )
            self.comm_sim = CommSimulator(
                bw_model,
                deadline_s=self.args.bw_deadline,
                tx_mode=getattr(self.args, "tx_mode", "all_or_nothing"),
            )
            logging.info(
                "[comm_sim] enabled: markov bandwidth (%s), deadline=%.1fs, seed=%d, tx_mode=%s"
                % (
                    getattr(self.args, "bw_profile", "uniform"),
                    self.args.bw_deadline,
                    self.args.bw_seed,
                    self.comm_sim.tx_mode,
                )
            )
            if self.comm_sim.tx_mode != "all_or_nothing":
                assert not getattr(self.args, "feddyn", False), (
                    "nested transmission not supported with feddyn yet"
                )
            logging.info(
                "[comm_sim] bandwidth spread: %s"
                % (
                    "continuous (jitter_sigma=%.2f)"
                    % getattr(self.args, "bw_jitter_sigma", 0.45)
                    if self.args.bw_profile == "tiered_continuous"
                    else "discrete states"
                )
            )
            # lazy caches for the nested path (see _build_tx_plan / _slice_client_update)
            self._tier_cfgs = None
            self._scratch_server = None
            # lazy caches for the shard path (see _allocate_shards)
            self._unit_bits = None
            self._unit_last_round = None

        # ---- client-SELECTION policy (E6b Stage 1, DIRECTION §3.2b) ----
        # Different decision surface from the assignment policy below: WHICH
        # clients are in the round, not which tier a selected client trains.
        # These tables are maintained unconditionally (cheap, and they are the
        # dead-slot / inclusion-gap metrics EXPERIMENTS.md now requires on every
        # run), but _client_sampling only consults a policy when one is set.
        self._min_tier_bits_cache = None
        if not hasattr(self, "_cls_counts"):
            self._cls_counts = None  # only set under a tiered bw profile above
        self._client_delivered_counts = {
            c: {} for c in range(self.args.client_num_in_total)
        }
        self._client_last_delivery_round = {
            c: -1 for c in range(self.args.client_num_in_total)
        }
        self._selection_policy = None
        sel_name = getattr(self.args, "selection_policy", "none")
        if sel_name != "none":
            from fedml_api.standalone.superfednas.comm_sim.policies import (
                SELECTION_POLICIES,
            )

            if self.comm_sim is None:
                raise ValueError(
                    "--selection_policy %r needs --comm_sim: without a bandwidth "
                    "model there is no notion of a dead slot, which is the whole "
                    "quantity these probes act on." % sel_name
                )
            self._selection_policy = SELECTION_POLICIES[sel_name]()
            logging.info(
                "[selection] selection_policy=%s active -- overriding the uniform "
                "client draw (DIRECTION §3.2b). NOTE: the Stage 1 probes read "
                "oracle bandwidth by design; they are diagnostics, not "
                "deployable policies." % sel_name
            )

        # ---- assignment policy ("policy_assign" sampler, DIRECTION §3.2,
        # retargeted 2026-07-22: "who trains what" under nested transmission) ----
        if self.args.subnet_dist_type == "policy_assign":
            from fedml_api.standalone.superfednas.comm_sim.policies import POLICIES

            policy_name = getattr(self.args, "assignment_policy", "none")
            if policy_name not in POLICIES:
                raise ValueError(
                    "--subnet_dist_type policy_assign requires --assignment_policy "
                    "in %s, got %r" % (list(POLICIES), policy_name)
                )
            if getattr(self.args, "tx_mode", "all_or_nothing") != "nested":
                # DeliveryFeedback/update() and _mark_tier_debt_served only fire
                # on the plain-nested transmit_plan() path below (round_forward
                # loop). all_or_nothing never reaches it (no truncation report),
                # and nested_shard's carrier branch + sub-floor shard fallback
                # both bypass it too -- either combination would silently leave
                # every assign()'d client's _pending entry never consumed (a
                # bandit that never learns, frozen at cold start; a Lyapunov
                # debt table that never marks tiers served) instead of a loud
                # failure. Caught during a pre-submission audit, not by a crash.
                raise ValueError(
                    "--subnet_dist_type policy_assign only supports --tx_mode "
                    "nested (delivery feedback isn't wired for %r) -- see the "
                    "comment above." % getattr(self.args, "tx_mode", None)
                )
            lam = getattr(self.args, "policy_lambda", None)
            if policy_name == "greedy":
                policy_obj = POLICIES[policy_name]()  # no lambda dependency (Lemma 1)
            else:
                policy_obj = POLICIES[policy_name](lam=lam)  # E3-gated, see policies/
            self.server_model.set_assignment_policy(policy_obj)
            self._policy = policy_obj  # direct reference: trainer calls .update()
            logging.info(
                "[policy] assignment_policy=%s active (lambda=%s)"
                % (policy_name, lam)
            )
            # Tier-level debt table (DIRECTION §3.2/§3.3): the Lyapunov virtual
            # queues Q_u(t), at TIER granularity (unlike the shard path's
            # per-BLOCK `_unit_last_round`). -1 = never served, so initial
            # staleness (round_num - last_round) is large for every tier.
            # Live reference stashed in sampler_args -- policy_assign_sample
            # reads it fresh every round without needing per-round rewiring.
            self._tier_last_round = {
                tier_id: -1 for tier_id in self.args.diverse_subnets
            }
            self.sampler_args["tier_debt"] = self._tier_last_round
            # Per-(client, tier) debt table (DIRECTION §3.2, added 2026-07-26
            # after the non-IID pilot found debt_by_tier's aggregate staleness
            # is data-blind: it can't tell a policy WHICH client refreshed a
            # tier, only WHEN. {client_idx: {tier_id: last_round}}. Only the
            # bandit consumes this (comm_sim/policies/bandit.py); Lyapunov's
            # per-tier-only design is left untouched on purpose.
            self._client_tier_last_round = {
                client_idx: {tier_id: -1 for tier_id in self.args.diverse_subnets}
                for client_idx in range(self.args.client_num_in_total)
            }
            self.sampler_args["client_tier_debt"] = self._client_tier_last_round
            # Cheap (config, loss) data-utility report (DIRECTION §3.2 /
            # FEEDBACK_REVIEW.md): every client's last local training loss,
            # always available regardless of bandwidth. {client_idx: {config,
            # loss}}. Live reference, same pattern as tier_debt above.
            self._data_utility = {}
            self.sampler_args["data_utility"] = self._data_utility

        self.wt_avg_sched_method = wt_avg_sched_method
        self.weighted_avg_scheduler = dict()
        self.weighted_avg_scheduler["Uniform"] = self.uniform_avg
        self.weighted_avg_scheduler[
            "maxnet_linear_all_subnet"
        ] = self.maxnet_linear_all_subnet
        self.weighted_avg_scheduler[
            "minnet_linear_all_subnet"
        ] = self.minnet_linear_all_subnet
        self.weighted_avg_scheduler[
            "maxnet_cos_all_subnet"
        ] = self.maxnet_cos_all_subnet
        self.weighted_avg_scheduler[
            "minnet_cos_all_subnet"
        ] = self.minnet_cos_all_subnet
        self.weighted_avg_scheduler[
            "maxnet_cos_all_subnet_sandwich"
        ] = self.maxnet_cos_all_subnet_sandwich
        self.weighted_avg_scheduler[
            "phased_maxnet_all_subnet"
        ] = self.phased_maxnet_all_subnet
        self.weighted_avg_scheduler[
            "multikd_phased_maxnet_all_subnet"
        ] = self.multikd_phased_maxnet_all_subnet

        # Wt Avg Scheduler Local Vars
        self.wt_avg_init = False
        self.init_maxnet_avg_wt = None
        self.mid_maxnet_avg_wt = None
        self.final_maxnet_avg_wt = None
        self.init_minnet_avg_wt = None
        self.final_minnet_avg_wt = None
        self.alpha = None
        self.weight_increment = None
        self.weight_increment1 = None
        self.weight_increment2 = None
        self.num_steps = None
        self.num_steps1 = None
        self.num_steps2 = None

        # Best Model Checkpointing
        self.prev_best = 0.0
        self.best_model_interval = self.args.best_model_freq

        # FedDyn alpha and server state
        self.feddyn = self.args.feddyn
        self.feddyn_alpha = args.feddyn_alpha
        self.model_param_len = 0
        if self.feddyn:
            self.model_param_len = model_vector(self.server_model.get_model_copy()).size()[
                0
            ]
        self.server_state = np.zeros(
            (self.args.client_num_in_total, self.model_param_len)
        ).astype("float32")
        self.weighted_feddyn_alpha = np.asarray(
            [
                self.train_data_local_num_dict[client_idx]
                for client_idx in range(self.args.client_num_in_total)
            ]
        )
        self.weighted_feddyn_alpha = (
            (self.weighted_feddyn_alpha / np.sum(self.weighted_feddyn_alpha))
            * self.feddyn_alpha
            * self.args.client_num_in_total
        )

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
            # ---- E6b: learned/probed client selection (DIRECTION §3.2b) ----
            # Placed AFTER the legacy draw on purpose. The np.random.seed above
            # reseeds the GLOBAL numpy stream, and every archived run's client
            # sequence depends on this exact call order -- consuming randomness
            # before it, or skipping the draw, silently changes WHICH clients
            # participate in every round while still producing plausible
            # accuracies, invalidating every cross-arm comparison in
            # results/README.md. So the legacy path always runs, untouched, and
            # the policy overrides its RESULT rather than replacing the draw.
            # Cost: one wasted np.random.choice per round. Guarded by
            # selftest_selection.py::test_legacy_selection_path_unchanged.
            if self._selection_policy is not None:
                client_indexes = self._policy_select_clients(
                    round_idx, client_num_in_total, num_clients
                )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _policy_select_clients(self, round_idx, client_num_in_total, num_clients):
        """Build the SelectionContext and delegate (E6b Stage 1).

        Candidates are ALL clients, not the legacy random subset -- the whole
        point is that the policy re-decides who is in the round. The probes read
        the true current-round bandwidth by design (they are diagnostics, see
        comm_sim/policies/selection.py); a deployable policy must use the lagged
        estimate, which is why both are wired here.
        """
        from fedml_api.standalone.superfednas.comm_sim.policies import (
            SelectionContext,
        )

        candidates = list(range(client_num_in_total))
        bw_true, bw_est = {}, {}
        if self.comm_sim is not None:
            lag = max(round_idx - 1, 0)
            for c in candidates:
                bw_true[c] = self.comm_sim.bw.bandwidth(c, round_idx)
                bw_est[c] = self.comm_sim.bw.bandwidth(c, lag)
        ctx = SelectionContext(
            bw_true_bps=bw_true,
            bw_est_bps=bw_est,
            min_tier_bits=self._min_tier_bits(),
            upload_window_s=(
                self.comm_sim.upload_window(0) if self.comm_sim is not None else None
            ),
            delivered_counts=self._client_delivered_counts,
            last_round_seen=self._client_last_delivery_round,
        )
        chosen = self._selection_policy.select(
            round_idx, candidates, num_clients, ctx
        )
        assert len(chosen) == num_clients and len(set(chosen)) == num_clients, (
            "selection policy returned %d entries (%d distinct), expected %d"
            % (len(chosen), len(set(chosen)), num_clients)
        )
        return list(chosen)

    def _min_tier_bits(self):
        """Payload of the smallest tier -- the dead-slot threshold. A client
        whose budget is below this delivers nothing whatever it is assigned."""
        if self.comm_sim is None:
            return None
        if self._min_tier_bits_cache is None:
            cfgs = self._tier_ladder_configs()
            self._min_tier_bits_cache = self.comm_sim.payload_bits(
                self.server_model.get_subnet(**cfgs[0])
            )
        return self._min_tier_bits_cache

    # ---- dynFedNAS v2: nested anytime transmission (Stage 1) ----
    def _tier_ladder_configs(self):
        """Tier ladder from diverse_subnets, ascending by payload (cached).
        Nestedness of the ladder is verified offline (verify_nesting.py)."""
        if self._tier_cfgs is None:
            subnets = self.sampler_args["diverse_subnets"]
            sized = []
            for key in subnets:
                cfg = subnets[key]
                model = self.server_model.get_subnet(**cfg)
                sized.append((self.comm_sim.payload_bits(model), cfg))
            sized.sort(key=lambda t: t[0])
            self._tier_cfgs = [cfg for _, cfg in sized]
        return self._tier_cfgs

    def _build_tx_plan(self, arch_config):
        """One client's shell ladder: its trained arch clipped by each tier.
        Payloads are sized with the same rule the simulator charges."""
        from fedml_api.standalone.superfednas.comm_sim.transmission import (
            build_clipped_plan,
        )

        return build_clipped_plan(
            arch_config,
            self._tier_ladder_configs(),
            lambda cfg: self.comm_sim.payload_bits(self.server_model.get_subnet(**cfg)),
        )

    def _slice_client_update(self, trained_model, beta_config):
        """Extract the delivered tier β ⊆ α from a trained α update.

        Merges the trained update into a scratch supernet copy (the client's
        avg_weight cancels in single-model _aggregate: sum=w*p, count=w), then
        cuts the β subnet out of it with the production get_subnet slicing.
        trained_model must already be on CPU (like _aggregate expects).
        """
        if self._scratch_server is None:
            self._scratch_server = copy.deepcopy(self.server_model)
        merged = self._aggregate([trained_model])
        self._scratch_server.set_model_params(merged)
        beta_model = self._scratch_server.get_subnet(**beta_config)
        beta_model.set_avg_wt(trained_model.avg_weight)
        return beta_model

    @staticmethod
    def _normalize_arch(config):
        """(d, e) as a comparable tuple, matching analysis/parse_runs.py's
        tier_of() convention. Needed because a SAMPLED/delivered config
        (from sample_subnet/get_subnet) is NOT the same structure as the
        raw --diverse_subnets dict it originated from: it gains a 'w' key
        and 'e' is expanded from a scalar into a per-block list -- naive
        dict/str equality between the two NEVER matches (caught 2026-07-24:
        the policy tally and tier-debt table were silently stuck at zero/
        never-updating because of exactly this, before Lyapunov's first
        real run -- see the [policy] round log this was built to catch)."""
        d = tuple(config.get("d", []))
        e = config.get("e")
        if isinstance(e, (list, tuple)):
            e = e[0] if len(set(e)) == 1 else None
        return (d, e)

    def _tier_id_for_config(self, config):
        """Map an architecture config back to its diverse_subnets key
        ("0".."3"), for tier-level debt tracking (comm_sim.policies). None if
        config doesn't match any configured tier (e.g. a non-tier arch)."""
        target = self._normalize_arch(config)
        for tier_id, cfg in self.args.diverse_subnets.items():
            if self._normalize_arch(cfg) == target:
                return tier_id
        return None

    def _mark_tier_debt_served(self, round_num, client_idx, plan, delivered_tier_idx):
        """DIRECTION §3.2/§3.3: a delivery reaching plan.tiers[delivered_tier_idx]
        refreshes THAT shell and every smaller one (nested shells are
        cumulative -- S subset M subset L subset XL), regardless of whether
        the delivery was native or cross-context (E3: cross-context is net
        beneficial, still counts as "served"). No-op if the config doesn't
        match a configured tier. Updates both the aggregate (Lyapunov) and
        per-client (bandit, added 2026-07-26) debt tables from the same loop
        -- one delivery event, two views of it."""
        for k in range(delivered_tier_idx + 1):
            tier_id = self._tier_id_for_config(plan.tiers[k][1])
            if tier_id is not None:
                self._tier_last_round[tier_id] = round_num
                self._client_tier_last_round[client_idx][tier_id] = round_num

    @staticmethod
    def _tier_name(subnet_info):
        """Clean wandb alias (tier_S..tier_XL) for the monitored uniform-e tier
        configs; None for arbitrary archs (which keep only the dict-keyed logs)."""
        e = subnet_info.get("e")
        if isinstance(e, (list, tuple)):
            e = e[0] if len(set(e)) == 1 else None
        return {0.1: "tier_S", 0.14: "tier_M", 0.18: "tier_L", 0.25: "tier_XL"}.get(e)

    def _log_tier_alias(self, subnet_info, test_acc, round_idx, tier_test_accs):
        tier = self._tier_name(subnet_info)
        if tier is not None:
            wandb.log(
                {f"Test/{tier}/Acc": test_acc, "round": round_idx}, step=round_idx
            )
            tier_test_accs[tier] = test_acc

    @staticmethod
    def _log_spread(tier_test_accs, round_idx):
        """Test/spread = XL - S accuracy: the Pareto-compression scalar."""
        if "tier_XL" in tier_test_accs and "tier_S" in tier_test_accs:
            wandb.log(
                {
                    "Test/spread": tier_test_accs["tier_XL"] - tier_test_accs["tier_S"],
                    "round": round_idx,
                },
                step=round_idx,
            )

    def _client_class_counts(self):
        """Per-client label histogram from the local train loaders (one pass at
        init; feeds the bandwidth-data correlation lever, E4)."""
        counts = {}
        for c, loader in self.train_data_local_dict.items():
            cnt = None
            for _, y in loader:
                y = y.view(-1).cpu().numpy().astype(int)
                m = int(y.max()) + 1
                if cnt is None:
                    cnt = np.zeros(m, dtype=np.int64)
                elif m > len(cnt):
                    cnt = np.concatenate([cnt, np.zeros(m - len(cnt), dtype=np.int64)])
                cnt += np.bincount(y, minlength=len(cnt))
            counts[c] = cnt if cnt is not None else np.zeros(1, dtype=np.int64)
        width = max(len(v) for v in counts.values())
        return {
            c: np.concatenate([v, np.zeros(width - len(v), dtype=np.int64)])
            for c, v in counts.items()
        }

    def _log_per_class_acc(self, round_idx):
        """Per-class test accuracy of each monitored tier probe on the global
        test set (--log_per_class). Feeds the class x method heatmap and the
        per-class-hole analysis under bandwidth-data correlation (E4). One
        compact parseable line per tier + wandb series.

        Returns {tier_name: per-class accuracy array} so _log_inclusion_gap can
        reuse it instead of running its own (and, before 2026-07-26, useless)
        per-client eval pass. {} when disabled."""
        if not getattr(self.args, "log_per_class", False):
            return {}
        per_class_acc = {}
        device = self.client_trainer.device
        for subnet_arch in self.args.ckpt_subnets:
            tier = self._tier_name(subnet_arch) or str(subnet_arch)
            # get_subnet() returns a ClientModel wrapper, not the nn.Module —
            # unwrap it (mirrors client_trainer.local_test's get_model() use).
            wrapped = self.server_model.get_subnet(**subnet_arch)
            model = wrapped.get_model()
            model.to(device)
            model.eval()
            correct, total = None, None
            with torch.no_grad():
                for x, y in self.test_global:
                    x, y = x.to(device), y.view(-1).to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    if correct is None:
                        num_cls = int(max(logits.shape[1], int(y.max()) + 1))
                        correct = torch.zeros(num_cls, device=device)
                        total = torch.zeros(num_cls, device=device)
                    total += torch.bincount(y, minlength=len(total)).float()
                    total_hits = y[pred == y]
                    correct += torch.bincount(
                        total_hits, minlength=len(correct)
                    ).float()
            acc = (correct / total.clamp(min=1)).cpu().numpy()
            per_class_acc[tier] = acc
            logging.info(
                "[per_class] round %d %s acc=%s"
                % (round_idx, tier, np.round(acc, 4).tolist())
            )
            wandb.log(
                {
                    **{
                        f"Test/{tier}/class_{k}_acc": float(acc[k])
                        for k in range(len(acc))
                    },
                    "round": round_idx,
                },
                step=round_idx,
            )
        return per_class_acc

    def _record_delivery(self, round_num, client_idx, delivered_tier_id):
        """One client's resolved outcome, for the selection layer's coverage
        tables and the dead-slot metric (EXPERIMENTS.md measurement rules,
        2026-07-26). Maintained on EVERY run, not only when a selection policy
        is active -- the blind baseline's numbers are exactly what the probes
        are compared against, so they have to be collected the same way.

        delivered_tier_id: the diverse_subnets key that arrived, or None if the
        client delivered nothing (a dead slot).
        """
        if delivered_tier_id is None:
            return
        counts = self._client_delivered_counts.setdefault(client_idx, {})
        counts[delivered_tier_id] = counts.get(delivered_tier_id, 0) + 1
        self._client_last_delivery_round[client_idx] = round_num

    def _log_selection_metrics(self, round_num, selected, delivered_clients):
        """[selection] round line: dead slots + coverage spread. This is the
        `fig_dead_slots` data and the only visibility into whether a selection
        policy is actually reclaiming wasted slots or just moving them."""
        dead = [c for c in selected if c not in delivered_clients]
        totals = [
            sum(self._client_delivered_counts.get(c, {}).values())
            for c in range(self.args.client_num_in_total)
        ]
        never = sum(1 for t in totals if t == 0)
        wandb.log(
            {
                "selection/dead_slots": len(dead),
                "selection/dead_slot_rate": len(dead) / max(len(selected), 1),
                "selection/coverage_min": min(totals) if totals else 0,
                "selection/coverage_max": max(totals) if totals else 0,
                "selection/clients_never_delivered": never,
                "round": round_num,
            },
            step=round_num,
        )
        logging.info(
            "[selection] round %d: selected=%d dead=%d (%s) coverage_min=%d "
            "coverage_max=%d never_delivered=%d"
            % (round_num, len(selected), len(dead), sorted(dead),
               min(totals) if totals else 0, max(totals) if totals else 0, never)
        )

    def _log_inclusion_gap(self, round_idx, per_class_acc):
        """Accuracy gap between the worst- and best-connectivity client groups
        (DIRECTION §2 secondary claim), as a CLASS-DISTRIBUTION-WEIGHTED read of
        the global test set:

            acc(group) = sum_c  p_group(c) * per_class_acc(c)

        where p_group(c) is the aggregate label distribution of the clients in
        that connectivity group. It answers "how well does the model serve the
        data held by poorly-connected clients", which is what the inclusion gap
        means, without needing a per-client test split.

        **Why not per-client local test sets (the obvious implementation, tried
        and discarded 2026-07-26 after the E6b smoke run job 103620):**
        `get_dataloader_CIFAR10` builds the test dataset WITHOUT `dataidxs`
        (fedml_api/data_preprocessing/cifar10/data_loader.py), so every client's
        `test_data_local_dict[c]` is the identical full CIFAR-10 test set --
        only the TRAIN set is partitioned. Evaluating per client therefore
        returns the same number for every group (the smoke printed
        {'poor': 0.1753, 'average': 0.1753, 'good': 0.1753} for every tier and
        round) while costing 20 clients x 4 tiers = 80 full test-set passes per
        checkpoint. The metric was structurally unmeasurable, not merely
        mis-scaled. Partitioning the test set would be the alternative, but
        `test_data_local_dict[0]` is what the mean_acc loop evaluates
        final_test_acc on, so changing it would silently redefine the headline
        metric of every future run and break comparability with results/.

        This version is free: it reuses the per-class accuracies
        `--log_per_class` already computes and the label histograms
        `_client_class_counts()` already built at init. It also has no call-
        ordering hazard, since it touches neither the model nor the dataloaders.

        Honest reading for the write-up: under corr=0 the groups hold similar
        class mixes, so the gap is near zero BY CONSTRUCTION and a policy cannot
        move it much -- the metric earns its keep under corr>0 (the
        digital-divide setting), which is exactly where DIRECTION §2 invokes it.
        """
        if not getattr(self.args, "log_inclusion_gap", False) or not per_class_acc:
            return
        tiers = getattr(getattr(self.comm_sim, "bw", None), "client_tiers", None)
        if tiers is None or self._cls_counts is None:
            return  # only meaningful under a tiered connectivity profile
        from fedml_api.standalone.superfednas.comm_sim import MarkovBandwidth

        # aggregate label distribution per connectivity group
        group_dist = {}
        for client_idx, tier in enumerate(tiers):
            counts = self._cls_counts.get(client_idx)
            if counts is None:
                continue
            grp = MarkovBandwidth.TIER_NAMES[tier]
            acc = group_dist.get(grp)
            group_dist[grp] = counts.astype(float) if acc is None else acc + counts
        group_dist = {
            g: v / max(v.sum(), 1.0) for g, v in group_dist.items() if v.sum() > 0
        }
        if "poor" not in group_dist or "good" not in group_dist:
            return

        for name, acc in per_class_acc.items():
            accs = {}
            for grp, dist in group_dist.items():
                n = min(len(dist), len(acc))
                accs[grp] = float((dist[:n] * acc[:n]).sum())
            gap = accs["good"] - accs["poor"]
            wandb.log(
                {
                    **{f"Inclusion/{name}/{g}": a for g, a in accs.items()},
                    f"Inclusion/{name}/gap": gap,
                    "round": round_idx,
                },
                step=round_idx,
            )
            logging.info(
                "[inclusion] round %d %s %s gap(good-poor)=%+.4f"
                % (round_idx, name,
                   {g: round(a, 4) for g, a in sorted(accs.items())}, gap)
            )

    def _allocate_shards(self, round_num, client_indexes):
        """Shard-carrier allocation for this round (nested_shard mode).

        Mechanism-default policy (pre-Lyapunov placeholder): pick the
        --shard_carriers clients with the highest *lagged* bandwidth (round
        r-1 — honest, no oracle), give each an estimated budget, and assign
        shard units (whole supernet blocks) greedily in coverage-debt order
        (least recently delivered first). A mis-estimate costs coverage this
        round, never correctness. Returns {client_idx: [unit ids]}.
        """
        from fedml_api.standalone.superfednas.comm_sim.sharding import (
            allocate_units,
            block_payload_bits,
        )

        if self._unit_bits is None:
            numels = {
                k: v.numel() for k, v in self.server_model.get_model_params().items()
            }
            self._unit_bits = block_payload_bits(
                numels, self.comm_sim.bytes_per_param
            )
            self._unit_last_round = {u: -1 for u in self._unit_bits}
            logging.info(
                "[comm_sim] shard units: %d blocks, %.1f MB total"
                % (
                    len(self._unit_bits),
                    sum(self._unit_bits.values()) / 8e6,
                )
            )
        num_carriers = getattr(self.args, "shard_carriers", 2)
        if num_carriers <= 0:
            return {}  # tables above stay initialized for the shard fallback
        bw_round = max(round_num - 1, 0)
        est = {
            cli: self.comm_sim.bw.bandwidth(cli, bw_round) for cli in client_indexes
        }
        carriers = sorted(est, key=est.get, reverse=True)[:num_carriers]
        budgets = {
            c: est[c] * self.comm_sim.upload_window(c) for c in carriers
        }
        debt_order = sorted(
            self._unit_bits, key=lambda u: (self._unit_last_round[u], u)
        )
        return allocate_units(self._unit_bits, budgets, debt_order)

    def _aggregate(self, w_locals):
        w_global_max_net = self.server_model.get_model_params()
        shared_param_count = dict()
        shared_param_sum = dict()
        for key, tensor in w_global_max_net.items():
            shared_param_count[key] = torch.zeros_like(tensor)
            shared_param_sum[key] = torch.zeros_like(tensor)

        # sum the local models.
        for w_local in w_locals:
            self.server_model.add_subnet(shared_param_sum, shared_param_count, w_local)

        # [Aggregate Logic]
        # shared_weights = zero tensors
        # update shared_weights from participating subnets by summing
        # w_supernet_{t+1} = shared_weights * 1/max(number_of_overlaps, 1) + (shared_weights is zero) w_supernet_{t}
        for key in w_global_max_net:
            w_global_max_net[key] = (
                shared_param_sum[key]
                * (
                    1.0
                    / (shared_param_count[key] + (shared_param_count[key] == 0).int())
                )
                + (shared_param_count[key] == 0) * w_global_max_net[key]
            )
        return w_global_max_net

    def train(self):
        if self.args.ckpt_subnets is None:
            self.args.ckpt_subnets = []
            for subnet_id in self.args.diverse_subnets:
                self.args.ckpt_subnets.append(self.args.diverse_subnets[subnet_id])
        for round_idx in range(self.start_round, self.args.comm_round):
            if round_idx >= self.best_model_interval:
                self.best_model_interval += self.args.best_model_freq
                self.prev_best = 0.0
            self.train_one_round(round_idx)
        if self.comm_sim is not None:
            s = self.comm_sim.summary()
            logging.info(
                "[comm_sim] run summary: drop_rate=%.3f max_subnet_failure_rate=%.3f "
                "wall_clock=%.1fs total_MB_sent=%.1f total_MB_dropped=%.1f "
                "total_MB_attempted=%.1f total_MB_wasted=%.1f waste_rate=%.3f"
                % (
                    s["comm/overall_drop_rate"],
                    s["comm/max_subnet_failure_rate"],
                    s["comm/final_wall_clock_s"],
                    s["comm/total_MB_sent"],
                    s["comm/total_MB_dropped"],
                    s["comm/total_MB_attempted"],
                    s["comm/total_MB_wasted"],
                    s["comm/overall_waste_rate"],
                )
            )
            for _k in (
                "comm/overall_drop_rate",
                "comm/max_subnet_failure_rate",
                "comm/final_wall_clock_s",
                "comm/total_MB_sent",
                "comm/total_MB_dropped",
            ):
                wandb.log({_k: s[_k]})
        if self.args.dry_run:
            return
        if self.args.wandb_watch:
            self.server_model.wandb_pass()
        self.server_model.save(
            "finished_checkpoint_data.pt", self.args.ofa_config, self.args.model,
        )

    def train_one_round(self, round_num, local_ep=None, **kwargs):
        client_indexes = self._client_sampling(
            round_num, self.args.client_num_in_total, self.args.client_num_per_round,
        )
        logging.info("client_indexes = " + str(client_indexes))
        self.server_model.set_cli_indices(client_indexes)
        # ---- dynFedNAS: give the capacity-aware sampler this round's bandwidth
        # picture before update_sample(), so the largest-subnet slot (and the
        # maxnet aggregation weight tied to it) only lands on eligible clients.
        # "oracle" assigns with this round's true bandwidth; "lagged" assigns
        # with the previous round's (realistic: the estimate can be stale).
        if self.comm_sim is not None and self.args.subnet_dist_type in (
            "capacity_aware",
            "policy_assign",
        ):
            bw_round = round_num
            if getattr(self.args, "bw_estimate", "oracle") == "lagged":
                bw_round = max(round_num - 1, 0)
            bw_now = {
                cli: self.comm_sim.bw.bandwidth(cli, bw_round)
                for cli in client_indexes
            }
            self.server_model.set_capacity_info(
                bw_now,
                self.comm_sim.deadline,
                self.sampler_args,
                bytes_per_param=self.comm_sim.bytes_per_param,
            )
        self.server_model.update_sample()
        avg_weights = self.weighted_avg_scheduler[self.wt_avg_sched_method](round_num)
        w_locals = []
        avg_gflops = 0
        avg_params = 0
        training_samples = []
        policy_tier_counts = None
        if self.args.subnet_dist_type == "policy_assign":
            # [policy] round summary (added 2026-07-24, before Lyapunov's
            # first real run): the only visibility into what the policy is
            # actually deciding round to round. Without this, a degenerate
            # policy (stuck always-max, or always-S) or a debt table that
            # stops updating would be invisible until final accuracy, with
            # no way to diagnose why.
            policy_tier_counts = {tid: 0 for tid in self.args.diverse_subnets}
        if self.comm_sim is not None:
            self.comm_sim.start_round(round_num)
        # nested_shard: designate carriers that train the full supernet and
        # upload assigned shard units of the starving region (collective upload)
        shard_assignment = {}
        if self.comm_sim is not None and self.comm_sim.tx_mode == "nested_shard":
            shard_assignment = self._allocate_shards(round_num, client_indexes)
        # clients that delivered SOMETHING this round -- the complement inside
        # client_indexes is the dead slots (E6b, see _log_selection_metrics)
        delivered_clients = set()

        for idx in range(self.args.client_num_per_round):
            # update dataset
            client_idx = client_indexes[idx]
            # DIRECTION §3.3 (delivery-aware aggregation weight, 2026-07-23):
            # fraction of this client's intended payload that actually arrived.
            # 1.0 unless overwritten below (nested truncation) -- explicitly
            # reset every iteration so a value from carrier/all_or_nothing
            # branches on a DIFFERENT client this same round never leaks in
            # (several branches below reuse local variable names like `plan`).
            delivery_completeness = 1.0

            self.client_trainer.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )

            client_model = self.server_model.sample_subnet(
                round_num, client_idx, idx, self.sampler_args
            )
            if policy_tier_counts is not None:
                tid = self._tier_id_for_config(client_model.model_config)
                if tid is not None:
                    policy_tier_counts[tid] += 1
            if self.args.dry_run:
                flops, _ = fp.profile(client_model.model, (1, 3, 32, 32))
                gflops = flops / 10e8
                avg_gflops += gflops
                params = sum(
                    p.numel()
                    for p in client_model.model.parameters()
                    if p.requires_grad
                )
                avg_params += params
                wandb.log(
                    {f"GFLOPS/idx:{idx}": gflops, "round": round_num,}, step=round_num,
                )
                wandb.log(
                    {f"Parameters/idx:{idx}": params, "round": round_num,},
                    step=round_num,
                )
                print(
                    f"Round: {round_num}, idx: {idx}, subnet arch: {client_model.model_config}, {gflops} gflops and {params} params"
                )
                continue

            # ---- communication simulation: decide the delivery BEFORE training ----
            # Delivery depends only on subnet size + bandwidth (not the trained
            # weights), so it can be decided here; a client that delivers nothing
            # is skipped entirely -> identical model result, no wasted compute.
            # all_or_nothing: the whole update fits or it is dropped (legacy).
            # nested(+shard): the upload is the client's clipped shell ladder;
            # the deadline truncates it and the largest complete tier survives.
            tx_report = None
            delivered_units = None
            if self.comm_sim is not None:
                carrier_units = shard_assignment.get(client_idx)
                if carrier_units:
                    # carrier: train the full supernet, upload ONLY the
                    # assigned shard units (its sampled subnet is overridden —
                    # the units require an XL-context update to exist)
                    from fedml_api.standalone.superfednas.comm_sim.transmission import (
                        ShardPlan,
                    )

                    client_model = self.server_model.get_subnet(
                        **self.server_model.max_subnet_arch()
                    )
                    plan = ShardPlan(
                        [(u, self._unit_bits[u]) for u in carrier_units]
                    )
                    # carriers are accounted with is_max_net=False: max_present/
                    # max_failed keep their sampler-slot meaning across arms
                    tx_report, delivered_units = self.comm_sim.transmit_shards(
                        round_num, client_idx, plan, is_max_net=False
                    )
                    if not delivered_units:
                        continue  # nothing fits: not trained, not aggregated
                    delivered_clients.add(client_idx)
                    self._record_delivery(
                        round_num, client_idx,
                        self._tier_id_for_config(client_model.model_config),
                    )
                elif self.comm_sim.tx_mode == "all_or_nothing":
                    is_max = self.server_model.is_max_net(client_model.model_config)
                    success, _ = self.comm_sim.transmit(
                        round_num, client_idx, client_model, is_max
                    )
                    if not success:
                        continue  # update would time out: not trained, not aggregated
                    delivered_clients.add(client_idx)
                    self._record_delivery(
                        round_num, client_idx,
                        self._tier_id_for_config(client_model.model_config),
                    )
                else:
                    is_max = self.server_model.is_max_net(client_model.model_config)
                    plan = self._build_tx_plan(client_model.model_config)
                    # fallback decision uses the LAGGED bandwidth (round r-1,
                    # same honesty as carrier selection): plan choices are
                    # ex-ante and may not use this round's true bandwidth.
                    # A mis-guess degrades gracefully either way; the actual
                    # delivery below is still truncated ex-post by the truth.
                    est_budget = self.comm_sim.bw.bandwidth(
                        client_idx, max(round_num - 1, 0)
                    ) * self.comm_sim.upload_window(client_idx)
                    if (
                        self.comm_sim.tx_mode == "nested_shard"
                        and not getattr(self.args, "no_shard_fallback", False)
                        and est_budget < plan.tiers[0][0]
                    ):
                        # sub-floor fallback: no complete tier fits this budget —
                        # upload whole blocks of the client's own subnet instead
                        # (collective aggregation rule; debt-ordered). Turns the
                        # residual all-or-nothing drops into partial contributions.
                        from fedml_api.standalone.superfednas.comm_sim.sharding import (
                            client_block_bits,
                        )
                        from fedml_api.standalone.superfednas.comm_sim.transmission import (
                            ShardPlan,
                        )

                        blk = client_block_bits(
                            client_model, self.comm_sim.bytes_per_param
                        )
                        order = sorted(
                            blk,
                            key=lambda u: (self._unit_last_round.get(u, -1), u),
                        )
                        fb_plan = ShardPlan([(u, blk[u]) for u in order])
                        tx_report, delivered_units = self.comm_sim.transmit_shards(
                            round_num, client_idx, fb_plan, is_max
                        )
                        if not delivered_units:
                            continue  # budget below a single block: nothing fits
                        delivered_clients.add(client_idx)
                        # sub-floor fallback delivers loose blocks, not a
                        # complete tier -- counts as "not a dead slot" but
                        # refreshes no tier's coverage.
                        self._record_delivery(round_num, client_idx, None)
                    else:
                        tx_report = self.comm_sim.transmit_plan(
                            round_num, client_idx, plan, is_max
                        )
                        # Needed by BOTH the policy feedback below and the
                        # coverage/dead-slot tables (which are maintained on
                        # every run, policy or not), so it is computed here
                        # rather than inside the policy_assign branch.
                        delivered_tier_id = (
                            self._tier_id_for_config(tx_report.delivered_config)
                            if tx_report.delivered_config is not None
                            else None
                        )
                        if self.args.subnet_dist_type == "policy_assign":
                            # Learned-policy feedback (DIRECTION §3.2, E6): must
                            # fire for EVERY resolved outcome, including total
                            # failure below -- a complete loss is exactly the
                            # signal a bandit needs to learn to avoid, not skip.
                            # Placed before the delivers_nothing() continue so
                            # neither path leaks an un-updated pending entry.
                            from fedml_api.standalone.superfednas.comm_sim.policies import (
                                DeliveryFeedback,
                            )

                            self._policy.update(DeliveryFeedback(
                                round_num=round_num,
                                client_idx=client_idx,
                                assigned_tier=self._tier_id_for_config(
                                    client_model.model_config
                                ),
                                delivered_tier=delivered_tier_id,
                                native_context=tx_report.completed_plan,
                                native_bits=(
                                    tx_report.bits_delivered
                                    if tx_report.completed_plan else 0.0
                                ),
                                cross_context_bits=(
                                    tx_report.bits_delivered
                                    if not tx_report.completed_plan else 0.0
                                ),
                            ))
                        if tx_report.delivers_nothing():
                            continue  # not even the smallest clipped tier fits
                        delivered_clients.add(client_idx)
                        self._record_delivery(
                            round_num, client_idx, delivered_tier_id
                        )
                        intended_bits = plan.total_bits()
                        if intended_bits > 0:
                            delivery_completeness = min(
                                tx_report.bits_delivered / intended_bits, 1.0
                            )
                        if (
                            self.args.subnet_dist_type == "policy_assign"
                            and tx_report.delivered_tier is not None
                        ):
                            self._mark_tier_debt_served(
                                round_num, client_idx, plan, tx_report.delivered_tier
                            )
                        if (
                            getattr(self.args, "nested_context_ablation", False)
                            and not tx_report.completed_plan
                        ):
                            # E3 ablation (DIRECTION §3.1a): discard cross-context
                            # deliveries (client trained alpha, truncated to a
                            # smaller beta) instead of aggregating the sliced
                            # beta-update. Isolates whether small tiers are
                            # "poisoned" by being aggregated partly from
                            # larger-context gradients -- compare this run's
                            # per-tier accuracy against a matched flag=False run
                            # (e.g. the existing arm-b nested baseline) with the
                            # same bandwidth seed. Skipped BEFORE training (the
                            # decision only needs config sizes, not weights), so
                            # this also saves compute on discarded clients.
                            continue

            maxnet_weight = avg_weights[idx]
            if (
                getattr(self.args, "maxnet_delivery_aware", False)
                and delivery_completeness < 1.0
            ):
                # DIRECTION §3.3: a client's MaxNet weight was assigned to its
                # PRE-round slot, before we knew whether the deadline would
                # truncate the delivery. Scale it down by how much of the
                # intended payload actually arrived, instead of applying the
                # full (possibly slot-max, beta up to 0.9 early in training)
                # weight to a small truncated slice. Off by default --
                # comparable to all prior results (v2_arm_b_nested_baseline.out
                # etc. all ran without this).
                maxnet_weight *= delivery_completeness
            client_model.set_avg_wt(maxnet_weight)
            cur_model_vec = None
            self.client_trainer.set_model(client_model)
            self.client_trainer.set_alpha(self.weighted_feddyn_alpha[client_idx])
            if self.feddyn:
                # send idx^th client' state
                server_model_copy = copy.deepcopy(self.server_model)
                server_model_copy.superimpose_vec(self.server_state[client_idx])
                superimposed_subnet = server_model_copy.get_subnet(
                    **client_model.model_config
                )
                superimposed_subnet_vec = model_vector(
                    superimposed_subnet, req_grad=False
                )
                self.client_trainer.send_client_state(superimposed_subnet_vec)
                server_model_copy = self.server_model.get_model_copy()
                cur_model_vec = model_dict_to_vector(
                    self._aggregate([client_model]), server_model_copy, req_grad=False,
                )
            cur_lr = self.args.lr
            if self.lr_scheduler is not None:
                cur_lr = self.lr_scheduler.get_lr(round_num)
            if self.args.verbose:
                logging.info(
                    f"Client_id: {client_idx} Round Number: {round_num} "
                    f"Subnet: {client_model.model_config} "
                    f" LR: {cur_lr}"
                    f" Counts: {self.server_model.cli_subnet_track[client_idx]}"
                )

            updated_cli_model = self.client_trainer.train(cur_lr, local_ep)
            sample_num = self.client_trainer.get_sample_number()
            training_samples.append(sample_num)
            if self.args.subnet_dist_type == "policy_assign":
                # cheap data-utility report (DIRECTION §3.2): always recorded,
                # regardless of whether this client's delivery later succeeds
                # or drops -- matches the "architecture-preference signal is
                # cheap and always transmittable" precision fix (FEEDBACK_REVIEW.md).
                self._data_utility[client_idx] = {
                    "config": client_model.model_config,
                    "loss": self.client_trainer.get_last_train_loss(),
                }
            # Update client state
            if self.feddyn:
                updated_cli_model_vec = model_dict_to_vector(
                    self._aggregate([updated_cli_model]),
                    self.server_model.get_model_copy(),
                )

                self.server_state[client_idx] += (
                    (updated_cli_model_vec - cur_model_vec).cpu().detach().numpy()
                )
            # move updated client model to cpu
            updated_cli_model.cpu()
            # shard carrier: aggregate only the delivered units of the XL update
            if delivered_units is not None:
                from fedml_api.standalone.superfednas.comm_sim.sharding import (
                    ShardedUpdate,
                )

                for unit in delivered_units:
                    self._unit_last_round[unit] = round_num
                if self.args.verbose:
                    role = "carrier" if carrier_units else "fallback"
                    logging.info(
                        "[comm_sim] %s %d delivered %d shard units"
                        % (role, client_idx, len(delivered_units))
                    )
                w_locals.append(ShardedUpdate(updated_cli_model, delivered_units))
                continue
            # nested transmission: the deadline delivered only tier β ⊂ α —
            # aggregate the β-slice of the trained weights (anytime property)
            if tx_report is not None and not tx_report.completed_plan:
                if self.args.verbose:
                    logging.info(
                        "[comm_sim] client %d truncated: trained %s, delivered "
                        "tier %d %s"
                        % (
                            client_idx,
                            client_model.model_config,
                            tx_report.delivered_tier,
                            tx_report.delivered_config,
                        )
                    )
                updated_cli_model = self._slice_client_update(
                    updated_cli_model, tx_report.delivered_config
                )
            w_locals.append(updated_cli_model)
        # Multiply weight with dataset size proportion to global dataset size
        # (len(w_locals) may be < client_num_per_round once comm_sim drops updates)
        if self.args.weight_dataset:
            total_training_samples = sum(training_samples)
            for idx in range(len(w_locals)):
                w_locals[idx].set_avg_wt(
                    w_locals[idx].avg_weight
                    * (float(training_samples[idx]) / total_training_samples)
                )
        if self.comm_sim is not None:
            comm_summary = self.comm_sim.end_round()
            if self.comm_sim.tx_mode == "nested_shard" and self._unit_bits:
                comm_summary["comm/shard_coverage"] = comm_summary.get(
                    "comm/shard_units_delivered", 0
                ) / len(self._unit_bits)
            if self.args.subnet_dist_type == "capacity_aware":
                comm_summary["capacity/num_clamped"] = (
                    self.server_model.cap_num_clamped
                )
                comm_summary["capacity/max_infeasible"] = (
                    self.server_model.cap_max_infeasible
                )
            for _k, _v in comm_summary.items():
                wandb.log({_k: _v, "round": round_num}, step=round_num)
            capacity_note = ""
            if self.args.subnet_dist_type == "capacity_aware":
                capacity_note = " clamped=%d max_infeasible=%d" % (
                    self.server_model.cap_num_clamped,
                    self.server_model.cap_max_infeasible,
                )
            # Cumulative totals on every round line so accuracy-vs-delivered-GB
            # curves can be joined on round from the .out alone (DIRECTION §5;
            # parse_runs.py matches this format).
            logging.info(
                "[comm_sim] round %d: participation=%.2f dropped=%d/%d "
                "max_subnet_present=%d max_subnet_failed=%d MB_sent=%.1f "
                "MB_delivered_total=%.1f MB_attempted_total=%.1f "
                "MB_wasted_total=%.1f wall_clock_s=%.1f%s"
                % (
                    round_num,
                    comm_summary["comm/participation_rate"],
                    comm_summary["comm/num_dropped"],
                    comm_summary["comm/num_selected"],
                    comm_summary["comm/max_subnet_present"],
                    comm_summary["comm/max_subnet_failed"],
                    comm_summary["comm/MB_sent_round"],
                    comm_summary["comm/MB_sent_total"],
                    comm_summary["comm/MB_attempted_total"],
                    comm_summary["comm/MB_wasted_total"],
                    comm_summary["comm/wall_clock_s"],
                    capacity_note,
                )
            )
        if policy_tier_counts is not None:
            mean_debt = {
                tid: round_num - last
                for tid, last in self._tier_last_round.items()
            }
            for tid, cnt in policy_tier_counts.items():
                wandb.log(
                    {f"policy/assigned_{tid}": cnt, "round": round_num},
                    step=round_num,
                )
                wandb.log(
                    {f"policy/debt_{tid}": mean_debt[tid], "round": round_num},
                    step=round_num,
                )
            logging.info(
                "[policy] round %d: assigned=%s debt=%s"
                % (round_num, policy_tier_counts, mean_debt)
            )
        if self.comm_sim is not None:
            self._log_selection_metrics(round_num, client_indexes, delivered_clients)
        if self.args.dry_run:
            avg_gflops /= self.args.client_num_per_round
            avg_params /= self.args.client_num_per_round
            wandb.log(
                {f"AVG GFLOPS": avg_gflops, "round": round_num,}, step=round_num,
            )
            wandb.log(
                {f"AVG Parameters": avg_params, "round": round_num,}, step=round_num,
            )
            print(
                f"Round {round_num}\tAVG_GFLOPS:{avg_gflops}\tAVG_Params:{avg_params}"
            )
            return
        supernet_aggregate = self._aggregate(w_locals)
        self.server_model.set_model_params(supernet_aggregate)
        # FedDyn Server Side model aggregation
        if self.feddyn:
            mean_client_states = np.mean(self.server_state, axis=0)
            self.server_model.sum_supernet_w_vec(mean_client_states)

        if self.args.wandb_watch and round_num % self.args.wandb_watch_freq == 0:
            self.server_model.wandb_pass()

        # test results
        # at last round
        if round_num == self.args.comm_round - 1:
            if self.args.efficient_test:
                self._efficient_local_test_on_all_clients(round_num)
            else:
                self._local_test_on_all_clients(round_num)
            _pc = self._log_per_class_acc(round_num)
            self._log_inclusion_gap(round_num, _pc)
        # per {frequency_of_the_test} round
        elif round_num % self.args.frequency_of_the_test == 0:
            if self.args.efficient_test:
                (_, subnet_test_acc_map,) = self._efficient_local_test_on_all_clients(
                    round_num
                )
            else:
                _, subnet_test_acc_map = self._local_test_on_all_clients(round_num)
            _pc = self._log_per_class_acc(round_num)
            mean_acc = 0.0
            for subnet_arch in self.args.ckpt_subnets:
                self.client_trainer.set_test_model(
                    self.server_model.get_subnet(**subnet_arch),
                )
                test_metrics = self.client_trainer.local_test(True)
                if self.args.dataset == 'ptb':
                    mean_acc += test_metrics["test_ppl"]
                else:
                    mean_acc += test_metrics["test_correct"] / test_metrics["test_total"]
            mean_acc /= len(self.args.ckpt_subnets)
            if self.args.dataset == 'ptb':
                wandb.log(
                    {f"Test/Mean/PPL": mean_acc, "round": round_num,}, step=round_num,
                )
                if mean_acc < self.prev_best:
                    self.prev_best = mean_acc
                    self.server_model.save(
                        f"best_checkpoint_supernet_{self.best_model_interval}.pt",
                        self.args.ofa_config,
                        self.args.model,
                    )
            else:
                wandb.log(
                    {f"Test/Mean/Acc": mean_acc, "round": round_num,}, step=round_num,
                )
                if mean_acc > self.prev_best:
                    self.prev_best = mean_acc
                    self.server_model.save(
                        f"best_checkpoint_supernet_{self.best_model_interval}.pt",
                        self.args.ofa_config,
                        self.args.model,
                    )
            self._log_inclusion_gap(round_num, _pc)
        # Checkpointing
        if round_num > 0 and round_num % self.args.model_checkpoint_freq == 0:
            self.server_model.save(
                f"finished_checkpoint_data_{round_num}.pt",
                self.args.ofa_config,
                self.args.model,
            )

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        avg_subnet_train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
        }

        avg_subnet_test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
        }
        subnet_test_acc_map = dict()
        subnet_train_acc_map = dict()
        tier_test_accs = dict()  # clean-alias tier accs for the S/M/L/XL probes
        for subnet_id in self.args.diverse_subnets:
            subnet_info = self.args.diverse_subnets[subnet_id]
            train_metrics = {
                "num_samples": [],
                "num_correct": [],
                "losses": [],
            }

            test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            for client_idx in range(self.args.client_num_in_total):
                """
                Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
                the training client number is larger than the testing client number
                """
                if self.test_data_local_dict[client_idx] is None:
                    continue

                self.client_trainer.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # set model first
                self.client_trainer.set_test_model(
                    self.server_model.get_subnet(
                        **self.args.diverse_subnets[subnet_id]
                    ),
                )

                # train data
                train_local_metrics = self.client_trainer.local_test(False)
                train_metrics["num_samples"].append(
                    copy.deepcopy(train_local_metrics["test_total"])
                )
                train_metrics["num_correct"].append(
                    copy.deepcopy(train_local_metrics["test_correct"])
                )
                train_metrics["losses"].append(
                    copy.deepcopy(train_local_metrics["test_loss"])
                )
                if self.args.verbose_test:
                    print("train stats", client_idx, train_local_metrics)

                # test data
                test_local_metrics = self.client_trainer.local_test(True)
                test_metrics["num_samples"].append(
                    copy.deepcopy(test_local_metrics["test_total"])
                )
                test_metrics["num_correct"].append(
                    copy.deepcopy(test_local_metrics["test_correct"])
                )
                test_metrics["losses"].append(
                    copy.deepcopy(test_local_metrics["test_loss"])
                )
                if self.args.verbose_test:
                    print("test stats", client_idx, test_local_metrics)

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_metrics["num_correct"]) / sum(
                train_metrics["num_samples"]
            )
            train_loss = sum(train_metrics["losses"]) / sum(
                train_metrics["num_samples"]
            )
            avg_subnet_train_metrics["num_correct"].append(train_acc)
            avg_subnet_train_metrics["losses"].append(train_loss)
            avg_subnet_train_metrics["num_samples"].append(1)

            # test on test dataset
            test_acc = sum(test_metrics["num_correct"]) / sum(
                test_metrics["num_samples"]
            )
            test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
            avg_subnet_test_metrics["num_correct"].append(test_acc)
            avg_subnet_test_metrics["losses"].append(test_loss)
            avg_subnet_test_metrics["num_samples"].append(1)

            stats = {
                "training_acc": train_acc,
                "training_loss": train_loss,
                "subnet": subnet_info,
            }

            wandb_subnet_log = dict()
            if "d" in subnet_info:
                wandb_subnet_log["d"] = subnet_info["d"]
            if "e" in subnet_info:
                wandb_subnet_log["e"] = subnet_info["e"]
            if "w" in subnet_info:
                wandb_subnet_log["w"] = subnet_info["w"]

            wandb.log(
                {f"Train/{wandb_subnet_log}/Acc": train_acc, "round": round_idx,},
                step=round_idx,
            )
            wandb.log(
                {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,},
                step=round_idx,
            )

            if self.args.verbose:
                logging.info(stats)

            subnet_train_acc_map[subnet_id] = stats["training_acc"]
            stats = {
                "test_acc": test_acc,
                "test_loss": test_loss,
                "subnet": subnet_info,
            }
            wandb.log(
                {f"Test/{wandb_subnet_log}/Acc": test_acc, "round": round_idx},
                step=round_idx,
            )
            wandb.log(
                {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,},
                step=round_idx,
            )
            self._log_tier_alias(subnet_info, test_acc, round_idx, tier_test_accs)
            logging.info(stats)
            subnet_test_acc_map[subnet_id] = stats["test_acc"]

        self._log_spread(tier_test_accs, round_idx)
        final_train_acc = sum(avg_subnet_train_metrics["num_correct"]) / sum(
            avg_subnet_train_metrics["num_samples"]
        )
        final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
            avg_subnet_train_metrics["num_samples"]
        )
        # test on test dataset
        final_test_acc = sum(avg_subnet_test_metrics["num_correct"]) / sum(
            avg_subnet_test_metrics["num_samples"]
        )
        final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
            avg_subnet_test_metrics["num_samples"]
        )
        final_stats = {
            "final_training_acc": train_acc,
            "final_training_loss": train_loss,
        }
        wandb.log({"Train/Acc": final_train_acc, "round": round_idx}, step=round_idx)
        wandb.log(
            {"Train/Loss": final_train_loss, "round": round_idx}, step=round_idx,
        )
        if self.args.verbose:
            logging.info(final_stats)

        final_stats = {
            "final_test_acc": test_acc,
            "final_test_loss": test_loss,
        }
        wandb.log({f"Test/Acc": final_test_acc, "round": round_idx}, step=round_idx)
        wandb.log({f"Test/Loss": final_test_loss, "round": round_idx}, step=round_idx)
        logging.info(final_stats)
        return subnet_train_acc_map, subnet_test_acc_map

    def _efficient_local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        avg_subnet_train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "ppl": [],
            "losses": [],
        }

        avg_subnet_test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "ppl": [],
            "losses": [],
        }

        subnet_train_acc_map = dict()
        subnet_test_acc_map = dict()
        tier_test_accs = dict()  # clean-alias tier accs for the S/M/L/XL probes

        for subnet_id in self.args.diverse_subnets:
            subnet_info = self.args.diverse_subnets[subnet_id]
            # set model first
            self.client_trainer.set_test_model(
                self.server_model.get_subnet(**subnet_info),
            )
            if not self.args.skip_train_test:
                # gather train partition accuracies
                train_metrics = {"num_samples": [], "num_correct": [], "ppl": [], "losses": []}
                for client_idx in range(self.args.client_num_in_total):
                    self.client_trainer.update_local_dataset(
                        0,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[0],
                        self.train_data_local_num_dict[client_idx],
                    )
                    # train data
                    train_local_metrics = self.client_trainer.local_test(False)
                    if self.args.dataset == 'ptb':
                        train_metrics["ppl"].append(
                            copy.deepcopy(train_local_metrics["test_ppl"])
                        )
                    else:
                        train_metrics["num_samples"].append(
                            copy.deepcopy(train_local_metrics["test_total"])
                        )
                        train_metrics["num_correct"].append(
                            copy.deepcopy(train_local_metrics["test_correct"])
                        )
                    train_metrics["losses"].append(
                        copy.deepcopy(train_local_metrics["test_loss"])
                    )
                    if self.args.verbose_test:
                        print("train stats", train_local_metrics)
            if self.args.dataset == "shakespeare":
                self.client_trainer.update_local_dataset(
                    0,
                    self.train_data_local_dict[0],
                    self.test_global,
                    self.train_data_local_num_dict[0],
                )
            test_metrics = dict()
            # test data
            test_local_metrics = self.client_trainer.local_test(True)
            if self.args.dataset == 'ptb':
                test_metrics["ppl"] = test_local_metrics["test_ppl"]
            else:
                test_metrics["num_samples"] = test_local_metrics["test_total"]
                test_metrics["num_correct"] = test_local_metrics["test_correct"]
            test_metrics["losses"] = test_local_metrics["test_loss"]

            if self.args.verbose_test:
                print("test stats", test_local_metrics)

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            if self.args.dataset == 'ptb':
                if not self.args.skip_train_test:
                    train_loss = sum(train_metrics["losses"]) / self.args.client_num_in_total
                    train_ppl = sum(train_metrics["ppl"]) / self.args.client_num_in_total
                    avg_subnet_train_metrics["ppl"].append(train_ppl)
                    avg_subnet_train_metrics["losses"].append(train_loss)
                    avg_subnet_train_metrics["num_samples"].append(1)

                    wandb_subnet_log = dict()
                    if "d" in subnet_info:
                        wandb_subnet_log["d"] = subnet_info["d"]
                    if "e" in subnet_info:
                        wandb_subnet_log["e"] = subnet_info["e"]

                    stats = {
                        "train_ppl": train_ppl,
                        "train_loss": train_loss,
                        "subnet": subnet_info,
                    }
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/PPL": train_ppl, "round": round_idx,},
                        step=round_idx,
                    )
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,},
                        step=round_idx,
                    )
                    logging.info(stats)
                    subnet_train_acc_map[subnet_id] = stats["train_ppl"]

                # test on test dataset
                test_ppl = test_metrics["ppl"]
                test_loss = test_metrics["losses"]
                avg_subnet_test_metrics["ppl"].append(test_ppl)
                avg_subnet_test_metrics["losses"].append(test_loss)
                avg_subnet_test_metrics["num_samples"].append(1)

                wandb_subnet_log = dict()
                if "d" in subnet_info:
                    wandb_subnet_log["d"] = subnet_info["d"]
                if "e" in subnet_info:
                    wandb_subnet_log["e"] = subnet_info["e"]
                if "w" in subnet_info:
                    wandb_subnet_log["w"] = subnet_info["w"]

                stats = {
                    "test_ppl": test_ppl,
                    "test_loss": test_loss,
                    "subnet": subnet_info,
                }
                wandb.log(
                    {f"Test/{wandb_subnet_log}/PPL": test_ppl, "round": round_idx},
                    step=round_idx,
                )
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,},
                    step=round_idx,
                )
                logging.info(stats)
                subnet_test_acc_map[subnet_id] = stats["test_ppl"]
            else:
                if not self.args.skip_train_test:
                    # test on train dataset
                    train_acc = sum(train_metrics["num_correct"]) / sum(
                        train_metrics["num_samples"]
                    )
                    train_loss = sum(train_metrics["losses"]) / sum(
                        train_metrics["num_samples"]
                    )
                    avg_subnet_train_metrics["num_correct"].append(train_acc)
                    avg_subnet_train_metrics["losses"].append(train_loss)
                    avg_subnet_train_metrics["num_samples"].append(1)

                    wandb_subnet_log = dict()
                    if "d" in subnet_info:
                        wandb_subnet_log["d"] = subnet_info["d"]
                    if "e" in subnet_info:
                        wandb_subnet_log["e"] = subnet_info["e"]
                    if "w" in subnet_info:
                        wandb_subnet_log["w"] = subnet_info["w"]

                    stats = {
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "subnet": subnet_info,
                    }
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Acc": train_acc, "round": round_idx,},
                        step=round_idx,
                    )
                    wandb.log(
                        {f"Train/{wandb_subnet_log}/Loss": train_loss, "round": round_idx,},
                        step=round_idx,
                    )
                    logging.info(stats)
                    subnet_train_acc_map[subnet_id] = stats["train_acc"]

                # test on test dataset
                test_acc = test_metrics["num_correct"] / test_metrics["num_samples"]
                test_loss = test_metrics["losses"] / test_metrics["num_samples"]
                avg_subnet_test_metrics["num_correct"].append(test_acc)
                avg_subnet_test_metrics["losses"].append(test_loss)
                avg_subnet_test_metrics["num_samples"].append(1)

                wandb_subnet_log = dict()
                if "d" in subnet_info:
                    wandb_subnet_log["d"] = subnet_info["d"]
                if "e" in subnet_info:
                    wandb_subnet_log["e"] = subnet_info["e"]
                if "w" in subnet_info:
                    wandb_subnet_log["w"] = subnet_info["w"]

                stats = {
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "subnet": subnet_info,
                }
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Acc": test_acc, "round": round_idx},
                    step=round_idx,
                )
                wandb.log(
                    {f"Test/{wandb_subnet_log}/Loss": test_loss, "round": round_idx,},
                    step=round_idx,
                )
                self._log_tier_alias(subnet_info, test_acc, round_idx, tier_test_accs)
                logging.info(stats)
                subnet_test_acc_map[subnet_id] = stats["test_acc"]

        self._log_spread(tier_test_accs, round_idx)
        if self.args.dataset == 'ptb':
            if not self.args.skip_train_test:
                # test on train dataset

                final_train_ppl = sum(avg_subnet_train_metrics["ppl"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )
                final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )

                final_stats = {
                    "final_train_ppl": final_train_ppl,
                    "final_train_loss": final_train_loss,
                }
                wandb.log({f"Train/PPL": final_train_ppl, "round": round_idx}, step=round_idx)
                wandb.log({f"Train/Loss": final_train_loss, "round": round_idx}, step=round_idx)
                logging.info(final_stats)

            # test on test dataset
            final_test_ppl = sum(avg_subnet_test_metrics["ppl"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )
            final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )

            final_stats = {
                "final_test_ppl": final_test_ppl,
                "final_test_loss": final_test_loss,
            }
            wandb.log({f"Test/PPL": final_test_ppl, "round": round_idx}, step=round_idx)
            wandb.log({f"Test/Loss": final_test_loss, "round": round_idx}, step=round_idx)
            logging.info(final_stats)
        else:
            if not self.args.skip_train_test:
                # test on train dataset
                final_train_acc = sum(avg_subnet_train_metrics["num_correct"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )
                final_train_loss = sum(avg_subnet_train_metrics["losses"]) / sum(
                    avg_subnet_train_metrics["num_samples"]
                )

                final_stats = {
                    "final_train_acc": final_train_acc,
                    "final_train_loss": final_train_loss,
                }
                wandb.log({f"Train/Acc": final_train_acc, "round": round_idx}, step=round_idx)
                wandb.log({f"Train/Loss": final_train_loss, "round": round_idx}, step=round_idx)
                logging.info(final_stats)

            # test on test dataset
            final_test_acc = sum(avg_subnet_test_metrics["num_correct"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )
            final_test_loss = sum(avg_subnet_test_metrics["losses"]) / sum(
                avg_subnet_test_metrics["num_samples"]
            )

            final_stats = {
                "final_test_acc": final_test_acc,
                "final_test_loss": final_test_loss,
            }
            wandb.log({f"Test/Acc": final_test_acc, "round": round_idx}, step=round_idx)
            wandb.log({f"Test/Loss": final_test_loss, "round": round_idx}, step=round_idx)
            logging.info(final_stats)
        return subnet_train_acc_map, subnet_test_acc_map

    def uniform_avg(self, round_num=None):
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(1)
        return subnet_flofa_avg_weights

    def maxnet_linear_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.weight_increment = (
                self.final_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps
            self.wt_avg_init = True
        round_num = min(round_num, self.num_steps)
        maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment * round_num
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.largest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def minnet_linear_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_minnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_minnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.weight_increment = (
                self.final_minnet_avg_wt - self.init_minnet_avg_wt
            ) / self.num_steps
            self.wt_avg_init = True
        round_num = min(round_num, self.num_steps)
        minnet_weight = self.init_minnet_avg_wt + self.weight_increment * round_num
        other_weight = (1 - minnet_weight) / (self.args.client_num_per_round - self.args.bottom_k_maxnet)
        minnet_weight /= self.args.bottom_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.smallest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = minnet_weight
                break
        return subnet_flofa_avg_weights

    def maxnet_cos_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.init_maxnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        maxnet_weight = self.init_maxnet_avg_wt * decayed
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.largest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def maxnet_cos_all_subnet_sandwich(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.init_maxnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        maxnet_weight = self.init_maxnet_avg_wt * decayed
        other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
        maxnet_weight /= self.args.top_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if idx == ((round_num + 1) % self.args.client_num_per_round):
                subnet_flofa_avg_weights[idx] = maxnet_weight
                break
        return subnet_flofa_avg_weights

    def minnet_cos_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_minnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.final_minnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.alpha = float(self.final_minnet_avg_wt) / self.init_minnet_avg_wt
            self.num_steps = self.args.weighted_avg_schedule["num_steps"]
            self.wt_avg_init = True

        round_num = min(round_num, self.num_steps)
        cos_decay = 0.5 * (1 + np.cos(np.pi * round_num / self.num_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        minnet_weight = self.init_minnet_avg_wt * decayed
        other_weight = (1 - minnet_weight) / (self.args.client_num_per_round - self.args.bottom_k_maxnet)
        minnet_weight /= self.args.bottom_k_maxnet
        subnet_flofa_avg_weights = []
        for i in range(self.args.client_num_per_round):
            subnet_flofa_avg_weights.append(other_weight)
        for idx in range(self.args.client_num_per_round):
            if (
                self.server_model.cli_indices[idx]
                in self.server_model.smallest_subnet_min_idx
            ):
                subnet_flofa_avg_weights[idx] = minnet_weight
                break
        return subnet_flofa_avg_weights

    def phased_maxnet_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.mid_maxnet_avg_wt = self.args.weighted_avg_schedule["mid"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps1 = self.args.weighted_avg_schedule["num_steps1"]
            self.num_steps2 = self.args.weighted_avg_schedule["num_steps2"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.mid_maxnet_avg_wt
            self.weight_increment1 = (
                self.mid_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps1
            self.weight_increment2 = (
                self.final_maxnet_avg_wt - self.mid_maxnet_avg_wt
            ) / self.num_steps2
            self.wt_avg_init = True
        if round_num < self.num_steps1:
            maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment1 * round_num
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights[idx] = maxnet_weight
                    break
        else:
            round_num = min(round_num, self.num_steps1 + self.num_steps2)
            cos_decay = 0.5 * (
                1 + np.cos(np.pi * round_num / self.num_steps1 + self.num_steps2)
            )
            decayed = (1 - self.alpha) * cos_decay + self.alpha
            maxnet_weight = self.init_maxnet_avg_wt * decayed
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights[idx] = maxnet_weight
                    break
        return subnet_flofa_avg_weights

    def multikd_phased_maxnet_all_subnet(self, round_num):
        if not self.wt_avg_init:
            self.init_maxnet_avg_wt = self.args.weighted_avg_schedule["init"]
            self.mid_maxnet_avg_wt = self.args.weighted_avg_schedule["mid"]
            self.final_maxnet_avg_wt = self.args.weighted_avg_schedule["final"]
            self.num_steps1 = self.args.weighted_avg_schedule["num_steps1"]
            self.num_steps2 = self.args.weighted_avg_schedule["num_steps2"]
            self.alpha = float(self.final_maxnet_avg_wt) / self.mid_maxnet_avg_wt
            self.weight_increment1 = (
                self.mid_maxnet_avg_wt - self.init_maxnet_avg_wt
            ) / self.num_steps1
            self.weight_increment2 = (
                self.final_maxnet_avg_wt - self.mid_maxnet_avg_wt
            ) / self.num_steps2
            self.wt_avg_init = True
        if round_num < self.num_steps1:
            maxnet_weight = self.init_maxnet_avg_wt + self.weight_increment1 * round_num
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
                subnet_flofa_avg_weights.append(maxnet_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights.pop(idx * 2)
                    break
        else:
            round_num = min(round_num, self.num_steps1 + self.num_steps2)
            cos_decay = 0.5 * (
                1 + np.cos(np.pi * round_num / self.num_steps1 + self.num_steps2)
            )
            decayed = (1 - self.alpha) * cos_decay + self.alpha
            maxnet_weight = self.init_maxnet_avg_wt * decayed
            other_weight = (1 - maxnet_weight) / (self.args.client_num_per_round - self.args.top_k_maxnet)
            maxnet_weight /= self.args.top_k_maxnet
            subnet_flofa_avg_weights = []
            for i in range(self.args.client_num_per_round):
                subnet_flofa_avg_weights.append(other_weight)
                subnet_flofa_avg_weights.append(maxnet_weight)
            for idx in range(self.args.client_num_per_round):
                if (
                    self.server_model.cli_indices[idx]
                    in self.server_model.largest_subnet_min_idx
                ):
                    subnet_flofa_avg_weights.pop(idx * 2)
                    break
        return subnet_flofa_avg_weights

    # FedDyn update server state variables
    def update_server_state(self, client_updates):
        model_delta = self.server_model.get_model_copy()
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)
        global_model = self.server_model.get_model_copy()
        for client_model in client_updates:
            for server_param, client_param, delta_param in zip(
                global_model.parameters(),
                client_model.parameters(),
                model_delta.parameters(),
            ):
                delta_param.data += (
                    client_param - server_param
                ) / self.args.client_num_in_total

        for state_param, delta_param in zip(
            self.server_model.get_server_state().parameters(), model_delta.parameters(),
        ):
            state_param.data -= self.feddyn_alpha * delta_param
