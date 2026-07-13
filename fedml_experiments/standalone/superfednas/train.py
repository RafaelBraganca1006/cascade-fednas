import argparse
import logging
import os
import random
import sys
import numpy as np
import torch

# --- dynfednas compat shim ---------------------------------------------------
# The public `ofa` package (0.1.0.post202012082159) still does
# `from torch._six import ...`, but torch >= 1.8 removed torch._six. Recreate the
# handful of names it needs so the pinned ofa build imports under modern torch.
# Must run before any ofa import (the ServerModel imports below trigger it).
if not hasattr(torch, "_six"):
    import types as _types
    import queue as _queue
    import collections.abc as _abc
    _six = _types.ModuleType("torch._six")
    _six.queue = _queue
    _six.string_classes = (str,)
    _six.int_classes = (int,)
    _six.container_abcs = _abc
    sys.modules["torch._six"] = _six
# --- end compat shim ---------------------------------------------------------

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_experiments.standalone.superfednas.parse_args import add_args
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import (
    load_partition_data_cifar100,
)
sys.path.append('../../../fedml_api/standalone/superfednas/elastic_nn/')
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.standalone.superfednas.elastic_nn.TCN import *
from fedml_api.standalone.superfednas.elastic_nn.TCN.word_cnn import *
from fedml_api.standalone.superfednas.elastic_nn.TCN.word_cnn.utils import load_ptb
from fedml_api.standalone.superfednas.elastic_nn.TCN.char_cnn.utils import load_shakespeare
from fedml_api.standalone.superfednas.elastic_nn.darts import genotypes
from fedml_api.standalone.superfednas.elastic_nn.darts.model import NetworkCIFAR
from fedml_api.standalone.superfednas import learning_rate_schedule as lrs
from fedml_api.standalone.superfednas.Server.ServerModel import (
    ServerResnet,
    ServerResnet_10_26,
    ServerMobilenetV3Large_32x32,
    ServerElasticTCNN,
    ServerElasticCharTCNN,
    ServerDarts,
)
from fedml_api.standalone.superfednas.Server.ServerTrainer import (
    FLOFA_Trainer,
    ClientSupernetTrainer,
    ClientSupernetPSTrainer,
)
from fedml_api.standalone.superfednas.Client.ClientTrainer import (
    SupernetTrainer,
    SubnetTrainer,
    MultinetTrainer,
    FedDynSubnetTrainer,
    PSSupernetTrainer,
)


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        raise ValueError("Not supported")

    elif dataset_name == "femnist":
        raise ValueError("Not supported")
    #renamed from fed_shakespeare to tf_shakespeare since this is the Tensorflow sourced version of shakespeare dataset
    elif dataset_name == "tf_shakespeare":
        raise ValueError("Not supported")

    elif dataset_name == "fed_cifar100":
        raise ValueError("Not supported")
    elif dataset_name == "stackoverflow_lr":
        raise ValueError("Not supported")
    elif dataset_name == "stackoverflow_nwp":
        raise ValueError("Not supported")

    elif dataset_name == "ILSVRC2012":
        raise ValueError("Not supported")

    elif dataset_name == "gld23k":
        raise ValueError("Not supported")

    elif dataset_name == "gld160k":
        raise ValueError("Not supported")
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.use_train_pkl
            )
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.use_train_pkl
            )
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
            )
         #Penn Tree Bank
        elif dataset_name == "ptb":
            #custom logic for PTB
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                n_words,
            ) = load_ptb(args.data_dir, args.partition_method, args.client_num_in_total, args.batch_size, args.device)
            ptb_dataset = [
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                n_words,
            ]
            args.n_words = n_words
            return ptb_dataset
        elif dataset_name == "shakespeare":
            #slightly more custom logic for shakespeare since:
            #doesn't support different client number and different partition methods
            #both are determined by the dataset generated by the leaf repository: https://github.com/TalwalkarLab/leaf
            data_loader = load_shakespeare
            (
                num_clients,
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.data_dir,
                args.batch_size,
                args.init_seed,
                args.device
            )

            #NOTE: OVERRIDING INPUT TOTAL NUMBER OF CLIENTS
            args.n_chars = class_num
            args.client_num_in_total = num_clients
        else:
            data_loader = load_partition_data_cifar10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
            )

    if centralized:
        train_data_local_num_dict = {
            0: sum(
                user_train_data_num
                for user_train_data_num in train_data_local_num_dict.values()
            )
        }
        train_data_local_dict = {
            0: [
                batch
                for cid in sorted(train_data_local_dict.keys())
                for batch in train_data_local_dict[cid]
            ]
        }
        test_data_local_dict = {
            0: [
                batch
                for cid in sorted(test_data_local_dict.keys())
                for batch in test_data_local_dict[cid]
            ]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid])
            for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {
            cid: combine_batches(test_data_local_dict[cid])
            for cid in test_data_local_dict.keys()
        }
        args.batch_size = args_batch_size

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def custom_ofa_net(model_name, num_classes, config, args):
    if args.bn_gamma_zero_init:
        print(
            f"******* Using Gamma zero initialization for last BN layers in Residual blocks"
        )
    if model_name == "ofaresnet50":
        raise ValueError(
            "ofaresnet50 is not currently supported. To be implemented supernetwork model"
        )
    elif model_name == "ofaresnet50_32x32":
        params = dict()
        params["n_classes"] = num_classes
        if config is not None:
            if "d" in config.keys() and config["d"] is not None:
                params["depth_list"] = config["d"]
            if "e" in config.keys() and config["e"] is not None:
                params["expand_ratio_list"] = config["e"]
            if "w" in config.keys() and config["w"] is not None:
                params["width_mult_list"] = config["w"]
        model = ServerResnet(
            params,
            args.subnet_dist_type,
            args.client_num_in_total,
            args.bn_gamma_zero_init,
        )
    elif model_name == "ofaresnet50_32x32_10_26":
        params = dict()
        params["n_classes"] = num_classes
        if config is not None:
            if "d" in config.keys() and config["d"] is not None:
                params["depth_list"] = config["d"]
            if "e" in config.keys() and config["e"] is not None:
                params["expand_ratio_list"] = config["e"]
            if "w" in config.keys() and config["w"] is not None:
                params["width_mult_list"] = config["w"]
        model = ServerResnet_10_26(
            params,
            args.subnet_dist_type,
            args.client_num_in_total,
            args.bn_gamma_zero_init,
        )
    elif model_name == "ofambv3large_32x32":
        params = dict()
        params["n_classes"] = num_classes
        if config is not None:
            if "d" in config.keys() and config["d"] is not None:
                params["depth_list"] = config["d"]
            if "e" in config.keys() and config["e"] is not None:
                params["expand_ratio_list"] = config["e"]
            if "ks" in config.keys() and config["ks"] is not None:
                params["ks_list"] = config["ks"]
        else:
            params["depth_list"] = [2, 3, 4]
            params["expand_ratio_list"] = [3, 4, 6]
            params["ks_list"] = [7]
        model = ServerMobilenetV3Large_32x32(
            params,
            args.subnet_dist_type,
            args.client_num_in_total,
            args.bn_gamma_zero_init,
        )
    elif model_name == 'tcn':
        params = dict()
        params["input_size"] = args.emsize
        params["output_size"] = num_classes
        params["num_channels"] = [args.nhid] * (args.levels - 1) + [args.emsize]
        params["kernel_size"] = args.ksize
        params["dropout"] = args.dropout
        params["emb_dropout"] = args.emb_dropout
        params["tied_weights"] = args.tied
        if config is not None:
            if "d" in config.keys() and config["d"] is not None:
                params["depth_list"] = config["d"]
            if "e" in config.keys() and config["e"] is not None:
                params["expand_ratio_list"] = config["e"]
        else:
            params["depth_list"] = [0, 1, 2]
            params["expand_ratio_list"] = [0.1, 0.2, 0.25, 0.5, 1.0]
        model = ServerElasticTCNN(
            params,
            args.subnet_dist_type,
            args.client_num_in_total,
        )
    elif model_name == 'chartcn':
        params = dict()
        params["input_size"] = args.emsize
        params["output_size"] = num_classes
        params["num_channels"] = [args.nhid] * (args.levels - 1) + [args.emsize]
        params["kernel_size"] = args.ksize
        params["dropout"] = args.dropout
        params["emb_dropout"] = args.emb_dropout
        if config is not None:
            if "d" in config.keys() and config["d"] is not None:
                params["depth_list"] = config["d"]
            if "e" in config.keys() and config["e"] is not None:
                params["expand_ratio_list"] = config["e"]
        else:
            params["depth_list"] = [0, 1, 2]
            params["expand_ratio_list"] = [0.1, 0.2, 0.25, 0.5, 1.0]
        model = ServerElasticCharTCNN(
            params,
            args.subnet_dist_type,
            args.client_num_in_total,
        )
    elif model_name == "darts":
        params = dict()
        params["C"] = args.init_channel_size
        params["num_classes"] = num_classes
        params["layers"] = args.darts_layers
        params["auxiliary"] = False
        params["genotype"] = genotypes.FedNAS_V1
        model = ServerDarts(
            params,
            args.subnet_dist_type,
            args.client_num_in_total,
        )
    else:
        raise ValueError(f"{model_name} is not currently supported.")

    return model


def create_model(args, model_name, output_dim, load_teacher=False):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    # Load pretrained model if provided (assumes correct wandb run path)
    if (
        load_teacher
        and args.teacher_ckpt_name is not None
        and args.teacher_run_path is not None
    ):
        print(f"loading the teacher model wandb run path: {args.teacher_run_path}")
        restored_model = wandb.restore(
            args.teacher_ckpt_name, run_path=args.teacher_run_path
        )
        print(f"loading model from local directory: {restored_model.name}")
        checkpoint = torch.load(restored_model.name)
        if "params" in checkpoint:
            model = custom_ofa_net(
                checkpoint["supernet_class"],
                output_dim,
                checkpoint["supernet_config"],
                args,
            )
            model.set_model_params(checkpoint["params"])
        else:
            model = custom_ofa_net(model_name, output_dim, args.ofa_config, args)
            model.set_model_params(checkpoint)
    elif args.model_ckpt_name is not None and args.run_path is not None:
        print(f"loading the model wandb run path: {args.run_path}")
        restored_model = wandb.restore(args.model_ckpt_name, run_path=args.run_path)
        print(f"loading model from local directory: {restored_model.name}")
        checkpoint = torch.load(restored_model.name)
        if "params" in checkpoint:
            model = custom_ofa_net(
                checkpoint["supernet_class"],
                output_dim,
                checkpoint["supernet_config"],
                args,
            )
            model.set_model_params(checkpoint["params"])
        else:
            model = custom_ofa_net(model_name, output_dim, args.ofa_config, args)
            model.set_model_params(checkpoint)
    else:
        model = custom_ofa_net(model_name, output_dim, args.ofa_config, args)
    return model


def custom_server_trainer(server_trainer_params):
    assert server_trainer_params is not None
    if args.cli_supernet:
        return ClientSupernetTrainer(**server_trainer_params)
    elif args.cli_supernet_ps:
        return ClientSupernetPSTrainer(**server_trainer_params)
    else:
        return FLOFA_Trainer(**server_trainer_params)


def custom_client_trainer(client_trainer_params):
    assert client_trainer_params is not None
    if args.cli_supernet:
        print(f"Using Client Supernet Trainer")
        return SupernetTrainer(**client_trainer_params)
    elif args.cli_supernet_ps:
        print(f"Using Client Supernet PS Trainer")
        return PSSupernetTrainer(**client_trainer_params)
    elif args.multi:
        print(f"Using Multinet Trainer")
        return MultinetTrainer(**client_trainer_params)
    elif args.feddyn:
        print(f"Using FedDyn Trainer")
        return FedDynSubnetTrainer(**client_trainer_params)
    else:
        print(f"Using Subnet Trainer")
        return SubnetTrainer(**client_trainer_params)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone"))
    args = parser.parse_args()
    logger.info(args)
    if args.kd_ratio > 0 and not args.multi:
        assert (
            args.teacher_ckpt_name is not None and args.teacher_run_path is not None
        ), "Specify Pretrained model for knowledge distillation"

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    print("cuda available", torch.cuda.is_available())
    logger.info(device)
    wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        # "none" (the sweep scripts' placeholder) must not become a literal
        # entity name — None lets wandb use the logged-in account's default
        entity=None if args.wandb_entity in ("none", "") else args.wandb_entity,
        config=args,
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)
    #torch.backends.cudnn.deterministic = True

    # load data
    args.device = device

    dataset = load_data(args, args.dataset)
    assert(args.top_k_maxnet+args.bottom_k_maxnet<=args.client_num_per_round, "Top k value too large for given number of clients per round")
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    args.num_classes = output_dim = dataset[7]
    server_model = create_model(args, model_name=args.model, output_dim=dataset[7])

    if args.wandb_watch:
        logging.warning("Watching model parameters")
        wandb.watch(
            server_model.model, log="parameters", log_freq=args.wandb_watch_freq,
        )
    client_trainer_params = dict()
    client_trainer_params["model"] = None
    client_trainer_params["device"] = device
    client_trainer_params["args"] = args
    server_trainer_params = dict()
    server_trainer_params["server_model"] = server_model
    server_trainer_params["dataset"] = dataset
    server_trainer_params["args"] = args
    teacher_model = None
    # Teacher model not supported for TCN super-networks
    if args.kd_ratio > 0:
        teacher_model = create_model(
            args, model_name=args.model, output_dim=dataset[7], load_teacher=True,
        )
        server_trainer_params["teacher_model"] = teacher_model
        client_trainer_params["teacher_model"] = teacher_model
    server_trainer_params["client_trainer"] = custom_client_trainer(
        client_trainer_params
    )
    # Setup configuration for learning rate schedule
    flofa_lr_scheduler = None
    if args.lr_schedule is not None and args.lr_schedule["type"] is not None:
        if args.lr_schedule["type"] == "linear":
            # check if user has provided initial weights for each network and if so, for all subnetworks
            initial_subnet_flofa_lrs = dict()
            peak_rounds = dict()
            for subnet_key in args.diverse_subnets:
                if "lr0" in args.diverse_subnets[subnet_key]:
                    initial_subnet_flofa_lrs[int(subnet_key)] = args.diverse_subnets[
                        subnet_key
                    ]["lr0"]
                else:
                    raise Exception(
                        "Only partial or no initial weights provided which isn't supported. Please provide initial weights for each subnetwork!"
                    )
                if "peak_round" in args.diverse_subnets[subnet_key]:
                    peak_rounds[int(subnet_key)] = args.diverse_subnets[subnet_key][
                        "peak_round"
                    ]

            final_lr = None
            largest_subnet_id = None
            end_round = args.comm_round
            # IT IS EXPECTED THAT "lrf" IS PROVIDED ONLY FOR THE LARGEST SUBNETWORK
            for subnet_key in args.diverse_subnets:
                if "lrf" in args.diverse_subnets[subnet_key]:
                    final_lr = args.diverse_subnets[subnet_key]["lrf"]
                    largest_subnet_id = int(subnet_key)
                    if "end_R" in args.diverse_subnets[subnet_key]:
                        end_round = args.diverse_subnets[subnet_key]["end_R"]
                    break

            if final_lr is None:
                raise Exception(
                    "Max subnetwork has no final learning rate 'lrf' provided!"
                )

            flofa_lr_scheduler = lrs.LinearLRSchedule(
                len(args.diverse_subnets.keys()),
                initial_subnet_flofa_lrs,
                final_lr,
                largest_subnet_id,
                peak_rounds,
                end_round,
                args.lr_schedule,
            )
        elif args.lr_schedule["type"] == "exponential":
            flofa_lr_scheduler = lrs.ExponentialLRSchedule(
                args.lr,
                args.lr_schedule["decay_rate"],
                args.lr_schedule["decay_freq"],
            )
        else:
            logging.warning(
                f"Unrecognized lr schedule/regime algorithm: {args.lr_schedule['type']}"
            )
    server_trainer_params["lr_scheduler"] = flofa_lr_scheduler

    # Setup configuration for weighted average (Assumes correct input if provided)
    server_trainer_params["wt_avg_sched_method"] = "Uniform"
    if (
        args.weighted_avg_schedule is not None
        and args.weighted_avg_schedule["type"] is not None
    ):
        server_trainer_params["wt_avg_sched_method"] = args.weighted_avg_schedule[
            "type"
        ]

    server_trainer = custom_server_trainer(server_trainer_params)
    server_trainer.train()
