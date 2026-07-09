from fedml_api.standalone.superfednas.Client.ClientTrainer.client_trainer import ClientTrainer
import torch
import logging
from torch import nn
import torch.nn.functional as F
import copy


# Note: note filtering non-trainable parameters
def model_vector(model, req_grad=True):
    if not req_grad:
        for p in model.parameters():
            p.requires_grad = False
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)


# TODO: Client state should be a vector for only the subnet (vectorize onto maxnet)
class FedDynSubnetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(FedDynSubnetTrainer, self).__init__(model, device, args, teacher_model)
        self.test_model = model
        self.alpha = args.feddyn_alpha
        self.max_norm = self.args.feddyn_max_norm
        # theta_(t-1)
        self.global_model_vector = None
        self.client_state = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def send_client_state(self, client_state):
        self.client_state = client_state

    def train(self, lr, local_ep, **kwargs):
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.train()

        # Set up global model vector for FedDyn
        self.global_model_vector = model_vector(
            copy.deepcopy(self.client_model.model), req_grad=False
        )
        self.global_model_vector = self.global_model_vector.to(self.device)
        if not self.args.use_bn:
            for m in self.client_model.modules():
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

        # train and update
        criterion = nn.CrossEntropyLoss(reduction="sum").to(self.device)
        cur_wd = self.args.wd
        if (
            self.client_model.is_max_net(self.client_model.model_config)
            and self.args.largest_subnet_wd
        ):
            cur_wd = self.args.largest_subnet_wd
        if (
            self.args.feddyn_no_wd_modifier is None
            or self.args.feddyn_no_wd_modifier is not True
        ):
            cur_wd += self.alpha
        if self.args.feddyn_override_wd > 0:
            cur_wd = self.args.feddyn_override_wd
        model_params = filter(lambda p: p.requires_grad, self.client_model.parameters())
        # Note feddyn alpha impacts weight decay
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=cur_wd,)
        else:
            optimizer = torch.optim.Adam(
                model_params, lr=lr, weight_decay=cur_wd, amsgrad=True,
            )

        self.client_model.to(self.device)
        self.client_model.train()

        epoch_loss = []
        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                self.client_model.zero_grad()
                log_probs = self.client_model.forward(x)
                if self.args.kd_ratio > 0:
                    with torch.no_grad():
                        soft_logits = self.teacher_model(x).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
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

                # Note divide loss by batch size
                loss /= list(labels.size())[0]

                # FedDyn Loss
                if self.client_state is not None:
                    model_vec = model_vector(self.client_model.model)
                    model_vec.to(self.device)
                    self.client_state = self.client_state.to(self.device)
                    loss += self.alpha * torch.sum(
                        model_vec * (-self.global_model_vector + self.client_state)
                    )
                # Note we zero grad optimizer
                optimizer.zero_grad()
                loss.backward()

                # to avoid nan loss
                # Note: max norm is 10 in feddyn codebase
                torch.nn.utils.clip_grad_norm_(
                    self.client_model.parameters(), self.max_norm
                )

                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if self.args.verbose:
                logging.info(
                    "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                        self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss),
                    )
                )
        if not self.args.use_bn:
            for m in self.client_model.modules():
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
        self.client_model.freeze()
        return self.client_model

    def test(self, dataset, args, **kwargs):
        model = self.test_model

        model.to(self.device)
        model.eval()
        if not args.use_bn:
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
