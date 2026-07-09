from fedml_api.standalone.superfednas.Client.ClientTrainer.client_trainer import ClientTrainer
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d
import torch
import logging
from torch import nn
from collections import defaultdict


class SupernetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(SupernetTrainer, self).__init__(model, device, args, teacher_model)

    # self.client_model is a supernet of type ClientModel
    def train(self, lr, local_ep, **kwargs):
        model = self.client_model
        if not self.args.use_bn:
            for m in model.modules():
                if isinstance(m, DynamicBatchNorm2d):
                    bn_lay = m.bn
                    bn_lay.eval()
                    bn_lay.weight.requires_grad = False
                    bn_lay.bias.requires_grad = False
                    bn_lay.running_mean.requires_grad = False
                    bn_lay.running_var.requires_grad = False
                    with torch.no_grad():
                        bn_lay.weight.fill_(1)
                        bn_lay.bias.fill_(0)
                        bn_lay.running_mean.fill_(0)
                        bn_lay.running_var.fill_(1)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=self.args.wd,
        )
        model.to(self.device)
        model.train()
        maxnet_optimizer = None
        if self.args.largest_subnet_wd:
            maxnet_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=self.args.largest_subnet_wd,
            )

        epoch_loss = defaultdict(list)
        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = defaultdict(list)
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                supernet = model.get_model()
                supernet.zero_grad()
                supernet.set_max_net()
                log_probs_max = supernet(x)
                loss = criterion(log_probs_max, labels)
                loss.backward()
                batch_loss["max"].append(loss.item())
                if self.args.optim_step_more and (
                    not self.args.largest_step_more and not self.args.largest_subnet_wd
                ):
                    optimizer.step()
                if self.args.largest_step_more:
                    if self.args.largest_subnet_wd:
                        maxnet_optimizer.step()
                    elif not self.args.optim_step_more:
                        optimizer.step()
                if self.args.inplace_kd:
                    # treat output of largest network as label for rest
                    log_probs_max.detach()
                    log_probs_max = log_probs_max.max(1).indices

                for _ in range(self.args.num_multi_archs):
                    # sample random subnet
                    random_arch = self.client_model.sample_random_subnet()
                    supernet.set_max_net(**random_arch)
                    log_probs = supernet(x)
                    if self.args.inplace_kd:
                        loss = criterion(log_probs, log_probs_max)
                    else:
                        loss = criterion(log_probs, labels)
                    loss.backward()
                    batch_loss["mid"].append(loss.item())
                    if self.args.optim_step_more:
                        optimizer.step()
                model.set_active_subnet(d=0, e=0.1)
                log_probs = model(x)
                if self.args.inplace_kd:
                    loss = criterion(log_probs, log_probs_max)
                else:
                    loss = criterion(log_probs, labels)
                loss.backward()
                batch_loss["min"].append(loss.item())
                optimizer.step()

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
            for m in model.modules():
                if isinstance(m, DynamicBatchNorm2d):
                    with torch.no_grad():
                        bn_lay = m.bn
                        assert (
                            bn_lay.weight.equal(torch.ones_like(bn_lay.weight)),
                            "BN weight param not all 1s",
                        )
                        assert (
                            bn_lay.bias.equal(torch.zeros_like(bn_lay.bias)),
                            "BN bias param not all 0s",
                        )
                        assert (
                            bn_lay.running_mean.equal(
                                torch.zeros_like(bn_lay.running_mean)
                            ),
                            "BN running mean param not all 0s",
                        )
                        assert (
                            bn_lay.running_var.equal(
                                torch.ones_like(bn_lay.running_var)
                            ),
                            "BN running var param not all 1s",
                        )
        return self.client_model

    def test(self, dataset, args, **kwargs):
        model = self.test_model
        model.to(self.device)
        model.eval()
        if not args.use_bn:
            for m in model.modules():
                if isinstance(m, DynamicBatchNorm2d):
                    with torch.no_grad():
                        bn_lay = m.bn
                        assert (
                            bn_lay.weight.equal(torch.ones_like(bn_lay.weight)),
                            "BN weight param not all 1s",
                        )
                        assert (
                            bn_lay.bias.equal(torch.zeros_like(bn_lay.bias)),
                            "BN bias param not all 0s",
                        )
                        assert (
                            bn_lay.running_mean.equal(
                                torch.zeros_like(bn_lay.running_mean)
                            ),
                            "BN running mean param not all 0s",
                        )
                        assert (
                            bn_lay.running_var.equal(
                                torch.ones_like(bn_lay.running_var)
                            ),
                            "BN running var param not all 1s",
                        )

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
