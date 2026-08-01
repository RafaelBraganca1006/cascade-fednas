import logging
import time

import torch
import wandb
from torch import nn


class FedNASAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num, client_num, model, device, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.device = device
        self.args = args
        self.model = model
        self.model_dict = dict()
        self.arch_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.train_acc_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.best_accuracy = 0
        self.best_accuracy_different_cnn_counts = dict()
        self.wandb_table = wandb.Table(columns=["Epoch", "Searched Architecture"])

        # ---- comm_sim (EXPERIMENTS.md E2c) -------------------------------
        # Same deadline harness the SuperFedNAS arms run under, so the claim
        # "the failure mode belongs to the static-communication assumption of
        # the whole FedNAS family, not to our anchor" is measured rather than
        # argued. A deadline drop means the SERVER DOES NOT USE the update, so
        # nothing in the MPI protocol has to change: the message still arrives,
        # we just refuse the payload. That is why this lives here and not in
        # FedNASServerManager.
        self.comm_sim = None
        self.round_idx = 0
        if getattr(args, "comm_sim", False):
            from fedml_api.standalone.superfednas.comm_sim.comm_simulator import (
                CommSimulator,
            )
            from fedml_api.standalone.superfednas.comm_sim.bandwidth import (
                MarkovBandwidth,
                assign_connectivity_tiers,
            )
            import numpy as np

            n = self.client_num
            counts = {c: np.ones(10) for c in range(n)}
            tiers = assign_connectivity_tiers(counts, corr=0.0, seed=args.bw_seed)
            bw = MarkovBandwidth.tiered_continuous(
                n, args.comm_round, tiers, seed=args.bw_seed,
                jitter_sigma=args.bw_jitter_sigma,
            )
            self.comm_sim = CommSimulator(bw, args.bw_deadline)
            # must precede the first _record(): _cur is created by start_round()
            self.comm_sim.start_round(0)
            logging.info(
                "[comm_sim] FedNAS harness on: deadline=%ss profile=tiered_continuous "
                "jitter=%s seed=%s clients=%d"
                % (args.bw_deadline, args.bw_jitter_sigma, args.bw_seed, n)
            )

    @staticmethod
    def _payload_bits(model_params, bytes_per_param=4):
        """Uplink size of one FedNAS update, counted the same way the
        SuperFedNAS arms count theirs (4 bytes per parameter)."""
        return sum(int(t.numel()) for t in model_params.values()) * bytes_per_param * 8

    def get_model(self):
        return self.model

    def add_local_trained_result(self, index, model_params, arch_params, sample_num, train_acc, train_loss):
        logging.info("add_model. index = %d" % index)
        if self.comm_sim is not None:
            # Mirrors CommSimulator.transmit() exactly -- same all_or_nothing()
            # rule, same accounting -- but takes a state_dict rather than a
            # ClientModel, which is what FedNAS ships up.
            from fedml_api.standalone.superfednas.comm_sim.transmission import (
                all_or_nothing,
            )

            payload = self._payload_bits(model_params)
            budget, bw = self.comm_sim.upload_budget_bits(index, self.round_idx)
            report = all_or_nothing(
                payload, budget, bw, self.comm_sim.upload_window(index)
            )
            self.comm_sim._record(index, False, payload, report)
            if report.delivers_nothing():
                # The update did not fit the round deadline. The uploaded flag
                # MUST still be set: check_whether_all_receive() waits on every
                # rank, so an un-flagged client deadlocks the run forever. This
                # is the one place the MPI structure constrains the design.
                self.flag_client_model_uploaded_dict[index] = True
                logging.info(
                    "[comm_sim] client %d DROPPED (%.1f MB payload > %.1f MB budget)"
                    % (index, payload / 8e6, budget / 8e6)
                )
                return
        self.model_dict[index] = model_params
        self.arch_dict[index] = arch_params
        self.sample_num_dict[index] = sample_num
        self.train_acc_dict[index] = train_acc
        self.train_loss_dict[index] = train_loss
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        if self.comm_sim is not None:
            summary = self.comm_sim.end_round()
            logging.info(
                "[comm_sim] round %d: delivered=%d/%d drop=%d MB_sent_total=%.1f "
                "MB_attempted_total=%.1f MB_wasted_total=%.1f"
                % (self.round_idx, len(self.model_dict), self.client_num,
                   summary["comm/num_dropped"],
                   summary["comm/MB_sent_total"],
                   summary["comm/MB_attempted_total"],
                   summary["comm/MB_wasted_total"])
            )
            self.round_idx += 1
            self.comm_sim.start_round(self.round_idx)
            if not self.model_dict:
                # Every client missed the deadline. Keep the previous global
                # model rather than crashing on model_list[0]; a round where
                # nobody delivers is a real outcome of a tight deadline, not an
                # error, and the vanilla SuperFedNAS arm handles it the same way.
                logging.info("[comm_sim] round %d: ALL DROPPED, model unchanged"
                             % (self.round_idx - 1))
                w = self.model.state_dict()
                return (w, self.model.arch_parameters()) \
                    if self.args.stage == "search" else w
        averaged_weights = self.__aggregate_weight()
        self.model.load_state_dict(averaged_weights)
        if self.args.stage == "search":
            averaged_alphas = self.__aggregate_alpha()
            self.__update_arch(averaged_alphas)
            return averaged_weights, averaged_alphas
        else:
            return averaged_weights

    def __update_arch(self, alphas):
        logging.info("update_arch. server.")
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def delivered_indices(self):
        """Clients whose update actually arrived this round. Without comm_sim
        this is every client, so the legacy path is unchanged."""
        return sorted(self.model_dict.keys())

    def __aggregate_weight(self):
        logging.info("################aggregate weights############")
        start_time = time.time()
        model_list = []
        for idx in self.delivered_indices():
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        # RENORMALISE over who actually delivered. The original divides by
        # all_train_data_num, the total across ALL clients, which is only
        # correct when everyone arrives. Under drops the weights would sum to
        # less than 1 and the aggregate would be silently scaled DOWN every
        # round -- looking like divergence and getting blamed on FedNAS rather
        # than on the harness. With no drops this is identical to the original.
        denom = sum(n for n, _ in model_list) or self.all_train_data_num
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / denom
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        logging.info("aggregate weights time cost: %d" % (end_time - start_time))
        return averaged_params

    def __aggregate_alpha(self):
        logging.info("################aggregate alphas############")
        start_time = time.time()
        alpha_list = []
        for idx in self.delivered_indices():
            alpha_list.append((self.sample_num_dict[idx], self.arch_dict[idx]))

        denom = sum(n for n, _ in alpha_list) or self.all_train_data_num
        (num0, averaged_alphas) = alpha_list[0]
        for index, alpha in enumerate(averaged_alphas):
            for i in range(0, len(alpha_list)):
                local_sample_number, local_alpha = alpha_list[i]
                w = local_sample_number / denom
                if i == 0:
                    alpha = local_alpha[index] * w
                else:
                    alpha += local_alpha[index] * w
        end_time = time.time()
        logging.info("aggregate alphas time cost: %d" % (end_time - start_time))
        return averaged_alphas

    def statistics(self, round_idx):
        # train acc
        train_acc_list = self.train_acc_dict.values()
        self.train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        logging.info('Round {:3d}, Average Train Accuracy {:.3f}'.format(round_idx, self.train_acc_avg))
        wandb.log({"Train Accuracy": self.train_acc_avg, "Round": round_idx})
        # train loss
        train_loss_list = self.train_loss_dict.values()
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)
        logging.info('Round {:3d}, Average Train Loss {:.3f}'.format(round_idx, train_loss_avg))
        wandb.log({"Train Loss": train_loss_avg, "Round": round_idx})

        # test acc
        logging.info('Round {:3d}, Average Validation Accuracy {:.3f}'.format(round_idx, self.test_acc_avg))
        wandb.log({"Validation Accuracy": self.test_acc_avg, "Round": round_idx})
        # test loss
        logging.info('Round {:3d}, Average Validation Loss {:.3f}'.format(round_idx, self.test_loss_avg))
        wandb.log({"Validation Loss": self.test_loss_avg, "Round": round_idx})

        logging.info("search_train_valid_acc_gap %f" % (self.train_acc_avg - self.test_loss_avg))
        wandb.log({"search_train_valid_acc_gap": self.train_acc_avg - self.test_loss_avg, "Round": round_idx})

    def infer(self, round_idx):
        self.model.eval()
        self.model.to(self.device)
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            start_time = time.time()
            test_correct = 0.0
            test_loss = 0.0
            test_sample_number = 0.0
            test_data = self.test_global
            # loss
            criterion = nn.CrossEntropyLoss().to(self.device)
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(self.device)
                    target = target.to(self.device)

                    pred = self.model(x)
                    if self.args.stage == "train":
                        loss = criterion(pred[0], target)
                        _, predicted = torch.max(pred[0], 1)
                    else:
                        loss = criterion(pred, target)
                        _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)
                logging.info("server test. round_idx = %d, test_loss = %s" % (round_idx, test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_loss_avg = test_loss

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))

    def record_model_global_architecture(self, round_idx):
        # save the structure
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        cnn_count = normal_cnn_count + reduce_cnn_count
        wandb.log({"cnn_count": cnn_count, "Round": round_idx})

        logging.info("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
        logging.info('genotype = %s', genotype)
        wandb.log({"genotype": str(genotype), "round_idx": round_idx})

        self.wandb_table.add_data(str(round_idx), str(genotype))
        wandb.log({"Searched Architecture": self.wandb_table})

        # save the cnn architecture according to the CNN count
        cnn_count = normal_cnn_count * 10 + reduce_cnn_count
        wandb.log({"searching_cnn_count(%s)" % cnn_count: self.test_acc_avg, "epoch": round_idx})
        if cnn_count not in self.best_accuracy_different_cnn_counts.keys():
            self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
            summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            wandb.run.summary[summary_key_cnn_structure] = self.test_acc_avg

            summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            wandb.run.summary[summary_key_best_cnn_structure] = round_idx
        else:
            if self.test_acc_avg > self.best_accuracy_different_cnn_counts[cnn_count]:
                self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
                summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_cnn_structure] = self.test_acc_avg

                summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_best_cnn_structure] = round_idx

        if self.test_acc_avg > self.best_accuracy:
            self.best_accuracy = self.test_acc_avg
            wandb.run.summary["best_valid_accuracy"] = self.best_accuracy
            wandb.run.summary["epoch_of_best_accuracy"] = round_idx
