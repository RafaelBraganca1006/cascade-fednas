from fedml_api.standalone.superfednas.Server.ServerTrainer.superfednas_trainer import FLOFA_Trainer
import logging
import copy
import wandb
import torch

from fedml_api.standalone.superfednas.Client.ClientModel import ClientModel
class ClientSupernetPSTrainer(FLOFA_Trainer):
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
        super(ClientSupernetPSTrainer, self).__init__(
            server_model,
            dataset,
            client_trainer,
            args,
            lr_scheduler,
            wt_avg_sched_method=wt_avg_sched_method,
            teacher_model=teacher_model,
            start_round=start_round,
        )
        self.client_trainer.set_model(ClientModel(self.server_model.model, None, None, True, self.server_model.random_subnet_arch, self.server_model.random_depth_subnet_arch))

    def _aggregate(self, w_locals):
        w_global_max_net = self.server_model.get_model_params()
        shared_param_sum = dict()
        for key, tensor in w_global_max_net.items():
            shared_param_sum[key] = torch.zeros_like(tensor)
        for k in w_global_max_net.keys():
            for i in range(0, len(w_locals)):
                shared_param_sum[k] += w_locals[i][k]
            w_global_max_net[k] = shared_param_sum[k] * (1 / len(w_locals))
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
        self.server_model.update_sample()
        w_locals = []
        training_samples = []

        for idx in range(self.args.client_num_per_round):
            # update dataset
            client_idx = client_indexes[idx]

            self.client_trainer.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )


            cur_lr = self.args.lr
            if self.lr_scheduler is not None:
                cur_lr = self.lr_scheduler.get_lr(round_num)
            if self.args.verbose:
                logging.info(
                    f"Client_id: {client_idx} Round Number: {round_num} "
                    f" LR: {cur_lr}"
                    f" Counts: {self.server_model.cli_subnet_track[client_idx]}"
                )
            self.client_trainer.set_model(ClientModel(copy.deepcopy(self.server_model.model), None, None, False, self.server_model.random_subnet_arch, self.server_model.random_depth_subnet_arch))
            updated_cli_model = self.client_trainer.train(cur_lr, local_ep, sample_depth_only=round_num<self.args.ps_depth_only)
            training_samples.append(self.client_trainer.get_sample_number())

            # move updated client model to cpu
            w_locals.append(updated_cli_model.state_dict())
        # Multiply weight with dataset size proportion to global dataset size
        if self.args.weight_dataset:
            total_training_samples = sum(training_samples)
            for idx in range(self.args.client_num_per_round):
                w_locals[idx].set_avg_wt(
                    w_locals[idx].avg_weight
                    * (float(training_samples[idx]) / total_training_samples)
                )

        supernet_aggregate = self._aggregate(w_locals)
        self.server_model.set_model_params(supernet_aggregate)

        if self.args.wandb_watch and round_num % self.args.wandb_watch_freq == 0:
            self.server_model.wandb_pass()

        # test results
        # at last round
        if round_num == self.args.comm_round - 1:
            if self.args.efficient_test:
                self._efficient_local_test_on_all_clients(round_num)
            else:
                self._local_test_on_all_clients(round_num)
        # per {frequency_of_the_test} round
        elif round_num % self.args.frequency_of_the_test == 0:
            if self.args.efficient_test:
                (_, subnet_test_acc_map,) = self._efficient_local_test_on_all_clients(
                    round_num
                )
            else:
                _, subnet_test_acc_map = self._local_test_on_all_clients(round_num)
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
            logging.info(stats)
            subnet_test_acc_map[subnet_id] = stats["test_acc"]

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
                logging.info(stats)
                subnet_test_acc_map[subnet_id] = stats["test_acc"]

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
