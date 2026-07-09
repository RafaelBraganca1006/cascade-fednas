import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm
from ofa.utils import flops_counter as fp

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_experiments.standalone.superfednas.parse_args import add_args_test
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import (
    load_partition_data_cifar100,
)
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from fedml_api.standalone.superfednas.Server.ServerModel import (
    ServerResnet,
    ServerResnet_10_26,
)


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        batch_size = 128  # temporary batch size
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
        model = ServerResnet(params, None, args.client_num_in_total, None, None,)
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
        model = ServerResnet_10_26(params, None, args.client_num_in_total, None, None,)
    else:
        raise ValueError(f"{model_name} is not currently supported.")

    return model


def test(model, dataset, device):
    model.to(device)
    model.eval()
    metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataset):
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = criterion(pred, target)

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics["test_correct"] += correct.item()
            metrics["test_loss"] += loss.item() * target.size(0)
            metrics["test_total"] += target.size(0)
    return metrics


# takes subnet arch
def get_flops(supernet, model_arch):
    subnet = supernet.get_subnet(**model_arch).model
    flops = fp.profile(subnet, (1, 3, 32, 32))
    del subnet
    return flops


def nas_flop(supernet, dataset, device, constraint, args):
    """Helpful internal functions"""

    def random_sample():
        debug_rate = 50
        counter = 0
        while True:
            sample_arch = supernet.random_subnet_arch()
            sample_eff, _ = get_flops(supernet, sample_arch)
            sample_eff /= 10e8
            if counter % debug_rate == 0:
                print(f"sampled efficiency flops: {sample_eff}")
            if sample_eff <= constraint:
                return sample_arch, sample_eff
            counter += 1

    if supernet is None:
        print("Supernet mismatch, no supernet received!")
        return

    """Run a single roll-out of regularized evolution to a fixed time budget."""
    max_time_budget = args.max_time_budget
    population_size = args.population_size
    mutation_numbers = int(round(args.mutation_ratio * population_size))
    parents_size = int(round(args.parent_ratio * population_size))
    print(
        f"NAS PARAMATERS\n Rounds:{max_time_budget}\n Pop Size:{population_size}\n Mut Num:{mutation_numbers}\n, Par Size:{parents_size}"
    )
    best_valids = [-100]
    population = []  # (validation, sample, latency) tuples
    child_pool = []
    efficiency_pool = []
    best_info = None
    print("Generate random population...")
    for i in range(population_size):
        sample, efficiency = random_sample()
        child_pool.append(sample)
        efficiency_pool.append(efficiency)
        print(f"got {i}th individual")
    print("Population generated!\n Calculating population accuracies")
    accs = []
    for ind, child in enumerate(child_pool):
        metric = test(supernet.get_subnet(**child).model, dataset, device)
        cur_acc = metric["test_correct"] / metric["test_total"]
        print(f"Individual {ind} has accuracy {cur_acc}")
        accs.append(cur_acc)
    print("Accuracies calculated! Setting up population tuples.")
    for i in range(mutation_numbers):
        population.append((accs[i], child_pool[i], efficiency_pool[i]))
    print("Start Evolution...")
    # After the population is seeded, proceed with evolving the population.
    for gen in tqdm(
        range(max_time_budget), desc="Searching with flop constraint (%s)" % constraint,
    ):
        """file1 = open("Iterations.txt", "a+")
        file1.write(str(iter) + ":" + str(datetime.now()) + " : " + str(constraint) + "\n")
        file1.close()"""
        parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
        acc = parents[0][0]
        print(
            f"Generation: {gen-1} Acc: {parents[0][0]}, Arch: {parents[0][1]}, GFLOPS:{parents[0][2]}"
        )

        if acc > best_valids[-1]:
            best_valids.append(acc)
            best_info = parents[0]
        else:
            best_valids.append(best_valids[-1])
        wandb.log(
            {
                f"NAS Evolution {constraint} GFLOPs Constraint": best_valids[-1],
                "generation": gen,
            },
        )

        population = parents
        child_pool = []
        efficiency_pool = []
        print("Mutating...")
        for i in range(mutation_numbers):
            new_sample = None
            time_count = 0
            while new_sample is None:
                """file1 = open("Iterations.txt", "a+")
                file1.write("Mutation : " + str(i) + " \n")
                file1.close()"""
                par_sample = population[np.random.randint(parents_size)][1]
                # Mutate
                new_sample = supernet.mutate_sample(par_sample, args.mutate_prob)
                efficiency, _ = get_flops(supernet, new_sample)
                efficiency /= 10e8
                if efficiency <= constraint or time_count > 50:
                    break
                else:
                    new_sample = None
                    time_count += 1
            child_pool.append(new_sample)
            efficiency_pool.append(efficiency)
        print("Mating (crossover)...")
        for i in range(population_size - mutation_numbers):
            new_sample = None
            time_count = 0
            while new_sample is None:
                """file1 = open("Iterations.txt", "a+")
                file1.write("Crossover : " + str(i) + "\n")
                file1.close()"""
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_sample = supernet.crossover_sample(par_sample1, par_sample2)
                efficiency, _ = get_flops(supernet, new_sample)
                efficiency /= 10e8
                if efficiency <= constraint or time_count > 50:
                    break
                else:
                    new_sample = None
                    time_count += 1
            child_pool.append(new_sample)
            efficiency_pool.append(efficiency)
        print("Calculating accuracies of children...")
        accs = []
        for child in child_pool:
            metric = test(supernet.get_subnet(**child).model, dataset, device)
            cur_acc = metric["test_correct"] / metric["test_total"]
            accs.append(cur_acc)
        for i in range(population_size):
            population.append((accs[i], child_pool[i], efficiency_pool[i]))
    return best_valids, best_info


"""
Notes: 
- Need the exact checkpoint model name to load
- Specify wandb parameters such as watch_wandb, wandb_entity
- Need to specify gpu
"""
if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args_test(argparse.ArgumentParser(description="FedAvg-standalone"))
    args = parser.parse_args()
    logger.info(args)
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
    if not args.multi_seed_test or (
        args.run_path is not None and args.model_ckpt_name is not None
    ):
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
        args.num_classes = output_dim = dataset[7]
        # create model.
        # Note if the model is DNN (e.g., ResNet), the training will be very slow.
        # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)

        # Loading previous model
        print(f"loading the model wandb run path: {args.run_path}")
        restored_model = wandb.restore(args.model_ckpt_name, run_path=args.run_path)
        print(f"loading model from local directory: {restored_model.name}")
        checkpoint = torch.load(restored_model.name)
        if "params" in checkpoint:
            server_model = custom_ofa_net(
                checkpoint["supernet_class"],
                dataset[7],
                checkpoint["supernet_config"],
                args,
            )
            server_model.set_model_params(checkpoint["params"])
        if "torch_rng_state" in checkpoint.keys():
            torch.set_rng_state(checkpoint["torch_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])

    # Test each subnet provided as input
    if args.test_subnets is not None:
        for subnet_arch in args.test_subnets:
            print("Testing ", subnet_arch)
            subnet = server_model.get_subnet(**subnet_arch)
            test_metrics = test(subnet.model, test_data_local_dict[0], device)
            print("Got ", test_metrics)
            wandb.log(
                {
                    f"Test/Acc/{subnet_arch}": test_metrics["test_correct"]
                    / test_metrics["test_total"],
                    "round": 0,
                }
            )

    # Perform NAS on each constraint
    if args.nas and args.nas_constraints is not None:
        wandb.define_metric("nas_constraint")
        wandb.define_metric("nas_best_accuracy", step_metric="nas_step")
        for ind, constraint in enumerate(args.nas_constraints):
            constraint = float(constraint)
            # best_acc = tuple(test accuracy, model, flop count)
            history, best_acc = nas_flop(
                server_model, test_data_local_dict[0], device, constraint, args
            )
            print(
                f"NAS Completed for constraint {constraint}...\n Best individual was {best_acc}\n Evolution history: {history}"
            )
            wandb.log(
                {
                    "nas_best_accuracy": best_acc[0],
                    "nas_best_arch": best_acc[1],
                    "nas_best_ind_GFLOPS": best_acc[2],
                    "nas_constraint": constraint,
                    "nas_step": ind,
                }
            )

    if (
        args.multi_seed_test
        and args.seed_list is not None
        and args.multi_seed_test_subnets is not None
    ):
        multi_seed_data = dict()
        for i, seed in enumerate(args.seed_list):
            multi_seed_data[seed] = dict()
            args.init_seed = seed
            print(f"Testing seed {args.init_seed}")
            # Set the random seed. The np.random seed determines the dataset partition.
            random.seed(args.init_seed)
            np.random.seed(args.init_seed)
            torch.manual_seed(args.init_seed)
            torch.cuda.manual_seed_all(args.init_seed)

            # load data
            dataset = load_data(args, args.dataset)
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

            # Loading previous model
            print(f"loading the model wandb run path: {args.multi_seed_run_paths[i]}")
            restored_model = wandb.restore(
                args.multi_seed_model_ckpt_names[i],
                run_path=args.multi_seed_run_paths[i],
            )
            print(f"loading model from local directory: {restored_model.name}")
            checkpoint = torch.load(restored_model.name)
            if "params" in checkpoint:
                server_model = custom_ofa_net(
                    checkpoint["supernet_class"],
                    dataset[7],
                    checkpoint["supernet_config"],
                    args,
                )
                server_model.set_model_params(checkpoint["params"])
            if "torch_rng_state" in checkpoint.keys():
                torch.set_rng_state(checkpoint["torch_rng_state"])
                np.random.set_state(checkpoint["numpy_rng_state"])
            # remove loaded model to allow models with same to be loaded
            os.remove(restored_model.name)
            if "gflop" not in multi_seed_data.keys():
                flop_dict = dict()
                for ind, subnet_arch in enumerate(args.multi_seed_test_subnets):
                    flops, _ = get_flops(server_model, subnet_arch)
                    flop_dict[ind] = flops / 10e8
                multi_seed_data["gflop"] = flop_dict
            for ind, subnet_arch in enumerate(args.multi_seed_test_subnets):
                print("Testing ", subnet_arch)
                subnet = server_model.get_subnet(**subnet_arch)
                test_metrics = test(subnet.model, test_data_local_dict[0], device)
                print("Got ", test_metrics)
                wandb.log(
                    {
                        f"Seed: {seed}/Subnet:{subnet_arch}/TestAcc": test_metrics[
                            "test_correct"
                        ]
                        / test_metrics["test_total"],
                        "seed": seed,
                    }
                )
                multi_seed_data[seed][ind] = (
                    test_metrics["test_correct"] / test_metrics["test_total"]
                )
        if args.save_file is not None:
            # process multi seed data and write to csv
            import csv

            with open(args.save_file, "w", newline="") as csvfile:
                multi_seed_writer = csv.writer(csvfile, delimiter=" ")
                for ind, subnet_arch in enumerate(args.multi_seed_test_subnets):
                    cur_accs = []
                    for seed in args.seed_list:
                        cur_accs.append(multi_seed_data[seed][ind])
                    multi_seed_writer.writerow(
                        [multi_seed_data["gflop"][ind]]
                        + [np.mean(cur_accs)]
                        + [np.std(cur_accs)]
                    )
            print("wrote to file")
