from fedml_api.standalone.superfednas.Client.ClientTrainer.client_trainer import ClientTrainer
import torch
import logging
from torch import nn
import torch.nn.functional as F
from collections import defaultdict


class MultinetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(MultinetTrainer, self).__init__(model, device, args, teacher_model)

    def _set_train(self):
        device = self.device
        for model in self.client_model["mid"]:
            model.to(device)
            model.train()
        if self.client_model["max"] is not None:
            self.client_model["max"].to(device)
            self.client_model["max"].train()
        if self.client_model["min"] is not None:
            self.client_model["min"].to(device)
            self.client_model["min"].train()

    @staticmethod
    def _disable_bn(model):
        if model is None:
            return
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False
                with torch.no_grad():
                    m.weight.fill_(1)
                    m.bias.fill_(0)
                    m.running_mean.fill_(0)
                    m.running_var.fill_(1)

    @staticmethod
    def _assert_bn_disabled(model):
        if model is None:
            return
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                with torch.no_grad():
                    assert (
                        m.weight.equal(torch.ones_like(m.weight)),
                        "BN weight param not all 1s",
                    )
                    assert (
                        m.bias.equal(torch.zeros_like(m.bias)),
                        "BN bias param not all 0s",
                    )
                    assert (
                        m.running_mean.equal(torch.zeros_like(m.running_mean)),
                        "BN running mean param not all 0s",
                    )
                    assert (
                        m.running_var.equal(torch.ones_like(m.running_var)),
                        "BN running var param not all 1s",
                    )

    def _get_optimizers(self, lr):
        optimizers = dict()
        assert self.args.client_optimizer == "sgd"
        optimizers["mid"] = []
        for model in self.client_model["mid"]:
            optimizers["mid"].append(
                torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=self.args.wd,
                )
            )
        if self.client_model["max"] is not None:
            optimizers["max"] = torch.optim.SGD(
                filter(
                    lambda p: p.requires_grad, self.client_model["max"].parameters(),
                ),
                lr=lr,
                weight_decay=self.args.largest_subnet_wd
                if self.args.largest_subnet_wd
                else self.args.wd,
            )

        if self.client_model["min"] is not None:
            optimizers["min"] = torch.optim.SGD(
                filter(
                    lambda p: p.requires_grad, self.client_model["min"].parameters(),
                ),
                lr=lr,
                weight_decay=self.args.wd,
            )

        return optimizers

    def zero_grad(self):
        for model in self.client_model["mid"]:
            model.zero_grad()
        if self.client_model["max"] is not None:
            self.client_model["max"].zero_grad()
        if self.client_model["min"] is not None:
            self.client_model["min"].zero_grad()

    # self.client_model is a dictionary {"max":type(ClientModel), "min":type(ClientModel), "middle":[ClientModel(s)]}
    def train(self, lr, local_ep, **kwargs):
        self._set_train()

        if not self.args.use_bn:
            for model in self.client_model["mid"]:
                MultinetTrainer._disable_bn(model)
            MultinetTrainer._disable_bn(self.client_model["min"])
            MultinetTrainer._disable_bn(self.client_model["max"])

        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizers = self._get_optimizers(lr)

        abs_no_kd = False
        if "max" not in optimizers:
            abs_no_kd = True

        epoch_loss = defaultdict(list)

        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = defaultdict(list)
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                self.zero_grad()

                if self.client_model["max"] is not None:
                    log_probs_max = self.client_model["max"].forward(x)
                    if not self.args.skip_train_largest:
                        loss = criterion(log_probs_max, labels)
                        loss.backward()
                        optimizers["max"].step()
                        batch_loss["max"].append(loss.item())

                    if self.args.inplace_kd:
                        # treat output of largest network as label for rest
                        log_probs_max.detach()
                        log_probs_max = log_probs_max.max(1).indices
                    elif self.args.kd_ratio > 0:
                        with torch.no_grad():
                            soft_logits = log_probs_max.detach()
                            soft_label = F.softmax(soft_logits, dim=1)
                averaged_loss = 0.0
                for ind, (model, optimizer) in enumerate(
                    zip(self.client_model["mid"], optimizers["mid"])
                ):
                    log_probs = model(x)
                    if self.args.inplace_kd and not abs_no_kd:
                        loss = criterion(log_probs, log_probs_max)
                    elif self.args.kd_ratio != 0 and not abs_no_kd:
                        if self.args.kd_type == "ce":
                            kd_loss = self.cross_entropy_loss_with_soft_target(
                                log_probs, soft_label
                            )
                        else:
                            kd_loss = F.mse_loss(log_probs, soft_logits)
                        loss = self.args.kd_ratio * kd_loss + (
                            1 - self.args.kd_ratio
                        ) * criterion(log_probs, labels)
                    else:
                        loss = criterion(log_probs, labels)
                    """if self.args.history_dist_ratio > 0 and not abs_no_kd:
                        #TODO: Implement proximal dist loss?
                        loss += self.proximal_dist_loss(
                            self.max_model, model, self.models_indexes[ind]
                        )"""
                    loss.backward()
                    optimizer.step()
                    averaged_loss += loss.item()
                averaged_loss /= min(len(self.client_model["mid"]), 1)
                batch_loss["mid"].append(averaged_loss)
                if self.client_model["mid"] is not None:
                    log_probs = self.client_model["mid"].forward(x)
                    if self.args.inplace_kd:
                        loss = criterion(log_probs, log_probs_max)
                    elif self.args.kd_ratio != 0 and not abs_no_kd:
                        if self.args.kd_type == "ce":
                            kd_loss = self.cross_entropy_loss_with_soft_target(
                                log_probs, soft_label
                            )
                        else:
                            kd_loss = F.mse_loss(log_probs, soft_logits)
                        loss = self.args.kd_ratio * kd_loss + (
                            1 - self.args.kd_ratio
                        ) * criterion(log_probs, labels)
                    else:
                        loss = criterion(log_probs, labels)
                    """if self.args.history_dist_ratio > 0 and not abs_no_kd:
                        # TODO: Implement proximal dist loss?
                        loss += self.proximal_dist_loss(
                            self.max_model,
                            self.min_model,
                            self.min_model_indexes,
                        )"""
                    loss.backward()
                    optimizers["min"].step()
                    batch_loss["min"].append(loss.item())

            for key in batch_loss:
                epoch_loss[key].append(sum(batch_loss[key]) / len(batch_loss[key]))
            if self.args.verbose:
                for key in epoch_loss:
                    logging.info(
                        "Client Index = {}\t subnet: {} \tEpoch: {}\tLoss: {:.6f}".format(
                            self.client_idx,
                            key,
                            epoch,
                            sum(epoch_loss[key]) / len(epoch_loss[key]),
                        )
                    )
        if not self.args.use_bn:
            for model in self.client_model["mid"]:
                MultinetTrainer._assert_bn_disabled(model)
            MultinetTrainer._assert_bn_disabled(self.client_model["min"])
            MultinetTrainer._assert_bn_disabled(self.client_model["max"])
        return self.client_model

    def test(self, dataset, args, **kwargs):
        model = self.test_model
        model.to(self.device)
        model.eval()
        if not args.use_bn:
            self._assert_bn_disabled(model)

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataset):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        return metrics
