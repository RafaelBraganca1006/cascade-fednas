from fedml_api.standalone.superfednas.Client.ClientTrainer.client_trainer import ClientTrainer
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d
import torch
import logging
from torch import nn
import torch.nn.functional as F

class PSSupernetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(PSSupernetTrainer, self).__init__(model, device, args, teacher_model)

    # self.client_model is a supernet of type ClientModel
    def train(self, lr, local_ep, sample_depth_only=False, **kwargs):
        model = self.client_model
        model.to(self.device)
        model.train()
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


        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                model.zero_grad()
                if self.args.kd_ratio > 0:
                    with torch.no_grad():
                        self.teacher_model.to(self.device)
                        soft_logits = self.teacher_model.forward(x).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                for i in range(self.args.num_multi_archs):
                    # sample random subnet
                    if i == 0 and self.args.PS_with_largest:
                        model.set_active_subnet({"d":[2,2,2,2], "e":0.25})
                    else:
                        if sample_depth_only:
                            random_arch =  self.client_model.sample_random_depth_subnet()
                        else:
                            random_arch = self.client_model.sample_random_subnet()
                        model.set_active_subnet(random_arch)
                    log_probs = model.forward(x)
                    if self.args.kd_ratio == 0:
                        loss = criterion(log_probs, labels)
                    else:
                        if self.args.kd_type == "ce":
                            kd_loss = self.cross_entropy_loss_with_soft_target(
                                log_probs, soft_label
                            )
                        else:
                            kd_loss = F.mse_loss(log_probs, soft_logits)
                        loss = self.args.kd_ratio * kd_loss + (
                            1 - self.args.kd_ratio
                        ) * criterion(log_probs, labels)
                    loss.backward()
                batch_loss.append(loss.item())
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(
                    self.client_model.parameters(), self.args.max_norm
                )
                optimizer.step()


            epoch_loss = sum(batch_loss) / len(batch_loss)
            if self.args.verbose:
                logging.info(
                    "Client Index = {}\t subnet: {} \tEpoch: {}\tLoss: {:.6f}".format(
                        self.client_idx,
                        epoch,
                        epoch_loss,
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
                pred = model.forward(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        return metrics
