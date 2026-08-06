import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml_api.standalone.fedavg.client import Client


class _PayloadView(object):
    """Adapter so ``CommSimulator.payload_bits`` can size a plain nn.Module.

    The simulator was written against the SuperFedNAS ``ClientModel`` wrapper,
    which exposes the network under ``.model``. FedAvg hands around a bare
    module, so this gives the simulator the shape it expects instead of
    duplicating the sizing rule -- the payload MUST be counted by the same code
    in both codebases or the byte axis is not comparable across arms, which is
    the whole point of running this baseline.
    """

    def __init__(self, model):
        self.model = model


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.comm_sim = None
        self._payload_view = None
        if getattr(args, "comm_sim", False):
            self._setup_comm_sim()

    # ---- dynFedNAS: the bandwidth simulator, wired into the ORIGINAL FedML
    # FedAvg loop (docs/EXPERIMENTS.md E10). Nothing about FedAvg is modified:
    # the aggregation, the client sampling and the local training are upstream
    # code. The only intervention is that a client whose update misses the
    # round deadline is not appended to w_locals.
    #
    # Why this is a legitimate baseline rather than an adaptation: it IS
    # FedAvg, running its own implementation. What we add is the channel, which
    # is the thing under study and which no FL codebase models.
    def _setup_comm_sim(self):
        from fedml_api.standalone.superfednas.comm_sim import (
            CommSimulator,
            MarkovBandwidth,
            assign_connectivity_tiers,
        )

        if getattr(self.args, "bw_profile", "uniform").startswith("tiered"):
            cls_counts = self._client_class_counts()
            client_tiers = assign_connectivity_tiers(
                cls_counts,
                corr=getattr(self.args, "bw_data_corr", 0.0),
                seed=self.args.bw_seed,
            )
            if self.args.bw_profile == "tiered_continuous":
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
            logging.info(
                "[comm_sim] connectivity tiers (corr=%.2f): %s"
                % (
                    getattr(self.args, "bw_data_corr", 0.0),
                    {
                        name: [c for c, t in enumerate(client_tiers) if t == i]
                        for i, name in enumerate(MarkovBandwidth.TIER_NAMES)
                    },
                )
            )
        else:
            bw_model = MarkovBandwidth.default(
                num_clients=self.args.client_num_in_total,
                num_rounds=self.args.comm_round,
                seed=self.args.bw_seed,
            )
        # all_or_nothing is the ONLY honest mode here and it is not a
        # simplification: FedAvg ships one fixed model with no subset
        # structure, so a prefix of its update is not a smaller valid model.
        # It CANNOT have anytime delivery -- that is the impossibility argument
        # in docs/THESIS.md 2b, and this arm is where it is measured.
        self.comm_sim = CommSimulator(
            bw_model,
            deadline_s=self.args.bw_deadline,
            tx_mode="all_or_nothing",
        )
        self._payload_view = _PayloadView(self.model_trainer.model)
        payload_mb = self.comm_sim.payload_bits(self._payload_view) / 8e6
        logging.info(
            "[comm_sim] enabled: markov bandwidth (%s), deadline=%.1fs, seed=%d, "
            "tx_mode=all_or_nothing, payload=%.1f MB (fixed -- one architecture)"
            % (
                getattr(self.args, "bw_profile", "uniform"),
                self.args.bw_deadline,
                self.args.bw_seed,
                payload_mb,
            )
        )

    def _client_class_counts(self):
        """Per-client label histogram, for the connectivity/data correlation
        lever. Mirrors the superfednas trainer's method so the tier assignment
        is produced by the same rule from the same partition."""
        counts = {}
        for c in range(self.args.client_num_in_total):
            loader = self.train_data_local_dict[c]
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

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            if self.comm_sim is not None:
                self.comm_sim.start_round(round_idx)

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # ---- dynFedNAS: does this client's update reach the server
                # before the deadline? Decided BEFORE training: the payload of a
                # fixed architecture does not depend on the trained values, so a
                # client that would miss the deadline is skipped entirely --
                # identical model result, no wasted compute. Same reasoning and
                # same placement as the superfednas trainer.
                if self.comm_sim is not None:
                    delivered, _ = self.comm_sim.transmit(
                        round_idx, client_idx, self._payload_view, is_max_net=False
                    )
                    if not delivered:
                        continue

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            if self.comm_sim is not None:
                self._log_comm_round(round_idx)

            # update global weights.
            # _aggregate normalises by the sum of sample counts over w_locals,
            # so dropped clients renormalise correctly with no change -- unlike
            # FedNASAggregator, which divides by the total over ALL clients and
            # would silently scale the aggregate down every round (the trap
            # recorded in docs/EXPERIMENTS.md E2c).
            if not w_locals:
                # every selected client missed the deadline: keep the previous
                # global model rather than aggregating an empty list.
                logging.info(
                    "[comm_sim] round %d: no client delivered, global model unchanged"
                    % round_idx
                )
            else:
                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def _log_comm_round(self, round_idx):
        """One [comm_sim] line per round, in the SAME format the superfednas
        trainer emits, so analysis/parse_runs.py reads this arm with no changes
        and the byte axis is directly comparable."""
        s = self.comm_sim.end_round()
        for _k, _v in s.items():
            wandb.log({_k: _v, "round": round_idx})
        logging.info(
            "[comm_sim] round %d: participation=%.2f dropped=%d/%d "
            "max_subnet_present=%d max_subnet_failed=%d MB_sent=%.1f "
            "MB_delivered_total=%.1f MB_attempted_total=%.1f "
            "MB_wasted_total=%.1f wall_clock_s=%.1f"
            % (
                round_idx,
                s["comm/participation_rate"],
                s["comm/num_dropped"],
                s["comm/num_selected"],
                s["comm/max_subnet_present"],
                s["comm/max_subnet_failed"],
                s["comm/MB_sent_round"],
                s["comm/MB_sent_total"],
                s["comm/MB_attempted_total"],
                s["comm/MB_wasted_total"],
                s["comm/wall_clock_s"],
            )
        )

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
