"""dynFedNAS — Search Phase runner (cluster script).

Consumes a trained supernet checkpoint (finished_checkpoint_data.pt, saved in
the wandb offline run dir) and produces the NAS-side evidence (DIRECTION §5):

  --mode tiers        evaluate the 4 monitored tier configs (sanity vs logs)
  --mode population   K random archs from the FULL elastic space, each
                      BN-recalibrated and tested -> the "whole space improved"
                      cloud (fig_population_cloud)
  --mode pareto       regularized evolution (nas_flop, unchanged from the
                      anchor) per GFLOPS constraint -> accuracy-vs-MACs front
                      (fig_pareto_fronts)

Output: one CSV (mode, arch JSON, gflops, acc) per invocation — feeds
analysis/parse and the figure notebook.

Usage (cluster, from fedml_experiments/standalone/superfednas):
  python search_phase.py --ckpt <path>/finished_checkpoint_data.pt \
      --mode population --pop 100 --out pop_nested_shard.csv
  python search_phase.py --ckpt ... --mode pareto \
      --constraints 0.2 0.4 0.8 1.6 3.2 --out pareto_nested_shard.csv

Notes:
  - Search fitness defaults to the test loader, matching SuperFedNAS's own
    protocol (comparability). --fitness train uses the global train loader as
    a proxy validation set instead (honest variant; report which one is used).
  - BN recalibration (set_running_statistics on a slice of train data) is ON
    by default for population/tiers: arbitrary subnets evaluated with supernet
    BN stats score artificially low.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

# --- torch._six compat shim (same as train.py; must precede any ofa import) ---
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

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "../../../fedml_api/standalone/superfednas/elastic_nn/")
    )
)

from test_superfednas_resnet import (  # noqa: E402
    custom_ofa_net,
    get_flops,
    load_data,
    nas_flop,
    test,
)
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics  # noqa: E402

TIER_CONFIGS = [
    {"d": [0, 0, 0, 0], "e": 0.1},
    {"d": [0, 1, 0, 1], "e": 0.14},
    {"d": [1, 1, 1, 1], "e": 0.18},
    {"d": [2, 2, 2, 2], "e": 0.25},
]


def cache_batches(loader, num_batches):
    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        batches.append((x, y))
    return batches


def eval_arch(server_model, arch, test_loader, device, recal_batches, max_batches=None):
    subnet = server_model.get_subnet(**arch)
    if recal_batches:
        set_running_statistics(subnet.model, recal_batches)
    loader = test_loader
    if max_batches is not None:
        loader = [b for i, b in enumerate(test_loader) if i < max_batches]
    metrics = test(subnet.model, loader, device)
    return metrics["test_correct"] / metrics["test_total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to finished_checkpoint_data.pt")
    ap.add_argument("--mode", required=True, choices=["tiers", "population", "pareto"])
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--pop", type=int, default=100, help="population mode: #archs")
    ap.add_argument("--constraints", type=float, nargs="+", default=[0.2, 0.4, 0.8, 1.6, 3.2],
                    help="pareto mode: GFLOPS constraints (upper edge of each band)")
    ap.add_argument("--nas_floor_ratio", type=float, default=0.75,
                    help="pareto mode: each constraint c is the band "
                         "[nas_floor_ratio*c, c], so the search returns the best "
                         "architecture AT the budget rather than anywhere below "
                         "it. 1.0 would demand an exact hit; 0.0 restores the "
                         "old ceiling-only behaviour.")
    ap.add_argument("--fitness", choices=["test", "train"], default="test",
                    help="pareto search fitness data (test = anchor protocol)")
    ap.add_argument("--bn_recal_batches", type=int, default=30,
                    help="train batches for BN recalibration (0 disables)")
    ap.add_argument("--fitness_batches", type=int, default=10,
                    help="test batches used as fitness proxy during pareto evolution "
                         "(matches SuperFedNAS original; 0 = full test set). "
                         "Winner is always re-evaluated on the full test set.")
    ap.add_argument("--seed", type=int, default=0)
    # evolution knobs (nas_flop reads these off args)
    ap.add_argument("--max_time_budget", type=int, default=20)
    ap.add_argument("--population_size", type=int, default=50)
    ap.add_argument("--mutation_ratio", type=float, default=0.5)
    ap.add_argument("--parent_ratio", type=float, default=0.25)
    ap.add_argument("--mutate_prob", type=float, default=0.1,
                     help="per-gene mutation probability inside nas_flop's "
                     "mutate_sample (matches train.py's default); missing "
                     "here crashed pareto mode with AttributeError.")
    # data plumbing (defaults match the training runs)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data_dir", default="./../../../data/cifar10")
    ap.add_argument("--partition_method", default="hetero")
    ap.add_argument("--partition_alpha", type=float, default=100.0)
    ap.add_argument("--client_num_in_total", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_data(args, args.dataset)
    train_global, test_global, num_classes = dataset[2], dataset[3], dataset[7]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    server_model = custom_ofa_net(
        ckpt["supernet_class"], num_classes, ckpt["supernet_config"], args
    )
    server_model.set_model_params(ckpt["params"])
    print("loaded %s (%s)" % (args.ckpt, ckpt["supernet_class"]))

    recal_batches = (
        cache_batches(train_global, args.bn_recal_batches)
        if args.bn_recal_batches > 0 else None
    )
    rows = []

    if args.mode == "tiers":
        for cfg in TIER_CONFIGS:
            gflops = get_flops(server_model, cfg)[0] / 10e8
            acc = eval_arch(server_model, cfg, test_global, device, recal_batches)
            print("tier %s: %.2f GFLOPS acc=%.4f" % (cfg, gflops, acc))
            rows.append(("tier", json.dumps(cfg), gflops, acc))

    elif args.mode == "population":
        for i in range(args.pop):
            arch = server_model.random_subnet_arch()
            gflops = get_flops(server_model, arch)[0] / 10e8
            acc = eval_arch(server_model, arch, test_global, device, recal_batches)
            print("[%d/%d] %.2f GFLOPS acc=%.4f" % (i + 1, args.pop, gflops, acc))
            rows.append(("population", json.dumps(arch, default=int), gflops, acc))

    elif args.mode == "pareto":
        fitness_loader = test_global if args.fitness == "test" else train_global
        fitness_batches = args.fitness_batches if args.fitness_batches > 0 else None
        fitness_proxy = (
            [b for i, b in enumerate(fitness_loader) if i < fitness_batches]
            if fitness_batches is not None else fitness_loader
        )
        print("pareto fitness proxy: %s batches (winner re-evaluated on full test set)"
              % (fitness_batches if fitness_batches else "all"))
        for constraint in args.constraints:
            _, best = nas_flop(server_model, fitness_proxy, device, constraint, args)
            if best is None:
                print("constraint %.2f: no feasible individual" % constraint)
                continue
            acc_fit, arch, gflops = best
            # report accuracy on the true test set (with BN recal), regardless
            # of what data drove the search
            acc = eval_arch(server_model, arch, test_global, device, recal_batches)
            print("constraint %.2f: %.2f GFLOPS fit=%.4f test=%.4f %s"
                  % (constraint, gflops, acc_fit, acc, arch))
            rows.append(("pareto@%.2f" % constraint, json.dumps(arch, default=int), gflops, acc))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "arch", "gflops", "test_acc"])
        w.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
