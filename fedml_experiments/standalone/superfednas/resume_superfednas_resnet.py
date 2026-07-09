import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import wandb
from yaml import load, dump

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_experiments.standalone.superfednas.parse_args import add_args
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import (
    load_partition_data_cifar100,
)
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from fedml_api.standalone.superfednas import learning_rate_schedule as lrs
from fedml_api.standalone.superfednas.Server.ServerModel import (
    ServerResnet,
    ServerResnet_10_26,
)
from fedml_api.standalone.superfednas.Server.ServerTrainer import (
    FLOFA_Trainer,
    ClientSupernetTrainer,
)
from fedml_api.standalone.superfednas.Client.ClientTrainer import (
    SupernetTrainer,
    SubnetTrainer,
    MultinetTrainer,
)


def load_args(args, prev_config):
    args.batch_size = prev_config["batch_size"]["value"]
    args.bn_gamma_zero_init = prev_config["bn_gamma_zero_init"]["value"]
    args.ci = prev_config["ci"]["value"]
    args.cli_supernet = prev_config["cli_supernet"]["value"]
    args.client_num_in_total = prev_config["client_num_in_total"]["value"]
    args.client_num_per_round = prev_config["client_num_per_round"]["value"]
    args.client_optimizer = prev_config["client_optimizer"]["value"]
    args.data_dir = prev_config["data_dir"]["value"]
    args.dataset = prev_config["dataset"]["value"]
    if args.diverse_subnets is None:
        args.diverse_subnets = prev_config["diverse_subnets"]["value"]
    args.efficient_test = prev_config["efficient_test"]["value"]
    args.epochs = prev_config["epochs"]["value"]
    args.init_seed = prev_config["init_seed"]["value"]
    args.inplace_kd = prev_config["inplace_kd"]["value"]
    args.kd_ratio = prev_config["kd_ratio"]["value"]
    args.kd_type = prev_config["kd_type"]["value"]
    args.largest_step_more = prev_config["largest_step_more"]["value"]
    args.largest_subnet_wd = prev_config["largest_subnet_wd"]["value"]
    args.lr = prev_config["lr"]["value"]
    args.lr_schedule = prev_config["lr_schedule"]["value"]
    args.model = prev_config["model"]["value"]
    args.model_checkpoint_freq = prev_config["model_checkpoint_freq"]["value"]
    args.multi = prev_config["multi"]["value"]
    args.multi_disable_rest_bn = prev_config["multi_disable_rest_bn"]["value"]
    args.multi_drop_largest = prev_config["multi_drop_largest"]["value"]
    args.num_multi_archs = prev_config["num_multi_archs"]["value"]
    args.ofa_config = prev_config["ofa_config"]["value"]
    args.partition_alpha = prev_config["partition_alpha"]["value"]
    args.partition_method = prev_config["partition_method"]["value"]
    args.reset_bn_sample_size = prev_config["reset_bn_sample_size"]["value"]
    args.reset_bn_stats = prev_config["reset_bn_stats"]["value"]
    args.reset_bn_stats_test = prev_config["reset_bn_stats_test"]["value"]
    args.skip_train_largest = prev_config["skip_train_largest"]["value"]
    args.subnet_dist_type = prev_config["subnet_dist_type"]["value"]
    args.teacher_ckpt_name = prev_config["teacher_ckpt_name"]["value"]
    args.teacher_run_path = prev_config["teacher_run_path"]["value"]
    args.use_bn = prev_config["use_bn"]["value"]
    args.warmup_init_lr = prev_config["warmup_init_lr"]["value"]
    args.warmup_rounds = prev_config["warmup_rounds"]["value"]
    args.wd = prev_config["wd"]["value"]
    args.weighted_avg_schedule = prev_config["weighted_avg_schedule"]["value"]
    if "best_model_freq" in prev_config.keys():
        args.best_model_freq = prev_config["best_model_freq"]["value"]


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

    elif dataset_name == "shakespeare":
        raise ValueError("Not supported")

    elif dataset_name == "fed_shakespeare":
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
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
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
            args.cli_subnet_track,
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
            args.cli_subnet_track,
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
        print(f"loading the teacher model wandb run path: {args.run_path}")
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
    else:
        return FLOFA_Trainer(**server_trainer_params)


def custom_client_trainer(client_trainer_params):
    assert client_trainer_params is not None
    if args.cli_supernet:
        print(f"Using Client Supernet Trainer")
        return SupernetTrainer(**client_trainer_params)
    elif args.multi:
        print(f"Using Multinet Trainer")
        return MultinetTrainer(**client_trainer_params)
    else:
        print(f"Using Subnet Trainer")
        return SubnetTrainer(**client_trainer_params)


"""
Notes: 
- Resume round and comm round need to be set precisely
- Need the exact checkpoint model name to load
- Specify wandb parameters such as watch_wandb, wandb_entity
- Need to specify gpu
- Specify frequency of test (NOTE THAT TESTING IS DONE WHENEVER ROUND % FREQ == 0)
- Need to specify verbose settings
- Warmup lr is not supported for resume edge case
- bn gamma initialization is not supported since only meant for resume
- Rest of parameters will be filled in using previous run info
"""
if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone"))
    args = parser.parse_args()

    # Load relevant parameters from prev run
    assert (
        args.run_path is not None
        and args.model_ckpt_name is not None
        and 0 < args.resume_round < args.comm_round
    ), "Need to specify run path, model name to resume from, have valid resume comm round and end round number"
    prev_config = args.custom_config
    if prev_config is None:
        prev_config_meta = wandb.restore(
            "config.yaml", run_path=args.run_path, root="./wandb/"
        )
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with open(prev_config_meta.name, "r") as stream:
            try:
                prev_config = load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)
    load_args(args, prev_config)
    print("loaded previous settings")
    logger.info(args)
    if args.kd_ratio > 0 and not args.multi:
        assert (
            args.teacher_ckpt_name is not None and args.teacher_run_path is not None
        ), "Specify Pretrained model for knowledge distillation"

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    logger.info(device)
    wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=args,
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    args.num_classes = output_dim = dataset[7]

    # Loading previous model
    print(f"loading the model wandb run path: {args.run_path}")
    restored_model = wandb.restore(args.model_ckpt_name, run_path=args.run_path)
    print(f"loading model from local directory: {restored_model.name}")
    checkpoint = torch.load(restored_model.name)
    args.cli_subnet_track = checkpoint["cli_subnet_track"]
    if "params" in checkpoint:
        server_model = custom_ofa_net(
            checkpoint["supernet_class"],
            dataset[7],
            checkpoint["supernet_config"],
            args,
        )
        server_model.set_model_params(checkpoint["params"])
    if args.wandb_watch:
        logging.warning("Watching model parameters")
        wandb.watch(
            server_model.model, log="parameters", log_freq=args.wandb_watch_freq
        )

    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])

    client_trainer_params = dict()
    client_trainer_params["model"] = None
    client_trainer_params["device"] = device
    client_trainer_params["args"] = args
    server_trainer_params = dict()
    server_trainer_params["server_model"] = server_model
    server_trainer_params["dataset"] = dataset
    server_trainer_params["args"] = args
    server_trainer_params["start_round"] = args.resume_round
    teacher_model = None
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
