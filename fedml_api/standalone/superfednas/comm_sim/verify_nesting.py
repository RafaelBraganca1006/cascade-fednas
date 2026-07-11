"""Nestedness verification for the configured tier ladder (Stage 1.1).

The nested anytime transmission mechanism (DIRECTION.md §3.1a) requires the
tier ladder to be *nested in supernet coordinates*: every parameter coordinate
touched by tier k must also be touched by tier k+1, so that shells
D_k = tier_k \\ tier_{k-1} are well-defined and a delivered shell prefix is a
valid smaller-tier update. OFA slicing is prefix-based so this should hold for
monotone (d, e) ladders — this script verifies it against the *production*
mapping (``add_subnet``) instead of assuming it.

Needs torch + the ofa package -> run on the cluster:
    python3 fedml_api/standalone/superfednas/comm_sim/verify_nesting.py \
        --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}'

For each tier it marks the touched coordinates by pushing an all-ones update
through ``add_subnet`` and reading the nonzero entries of the count tensors,
then checks mask_k ⊆ mask_{k+1} and prints the shell-size table (params and MB
per shell) that transmission.TransmissionPlan will use.
"""

import argparse
import json
import os
import sys

# make the repo root importable regardless of where the script is invoked from,
# and elastic_nn/ for the bare `darts` import the ServerModel package pulls in
# (same trick as train.py line ~34)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(_here, "..", "elastic_nn")))

import torch

# same torch._six compat shim as train.py — the pinned public ofa build still
# imports torch._six, removed in torch >= 1.8. Must run before any ofa import.
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

# import the concrete module directly — the ServerModel package __init__ pulls
# in every server model (darts, mbv3, ...) and their extra deps, none needed here
from fedml_api.standalone.superfednas.Server.ServerModel.server_resnet_32x32_10_26 import (  # noqa: E501
    ServerResnet_10_26,
)


def touched_mask(server_model, config):
    """Coordinate mask (dict key -> bool tensor) touched by one tier,
    computed with the production add_subnet mapping."""
    w_global = server_model.get_model_params()
    param_sum = {k: torch.zeros_like(v) for k, v in w_global.items()}
    param_count = {k: torch.zeros_like(v) for k, v in w_global.items()}
    client_model = server_model.get_subnet(**config)
    client_model.set_avg_wt(1.0)
    server_model.add_subnet(param_sum, param_count, client_model)
    return {k: (param_count[k] != 0) for k in param_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diverse_subnets",
        type=json.loads,
        default=json.loads(
            '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},'
            '"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}'
        ),
        help="Tier ladder to verify (same JSON as the training runs).",
    )
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--bytes_per_param", type=int, default=4)
    args = parser.parse_args()

    server_model = ServerResnet_10_26(
        {"n_classes": args.n_classes}, "static", num_cli_total=1,
    )

    tiers = sorted(args.diverse_subnets.items(), key=lambda kv: int(kv[0]))
    configs = [cfg for _, cfg in tiers]
    masks = [touched_mask(server_model, cfg) for cfg in configs]
    sizes = [sum(int(m.sum()) for m in mask.values()) for mask in masks]

    print("tier ladder:")
    ok = True
    prev_size = 0
    for k, (cfg, size) in enumerate(zip(configs, sizes)):
        shell = size - prev_size
        print(
            "  tier %d %s: %d params (%.2f MB) | shell D%d: %d params (%.2f MB)"
            % (
                k, cfg, size, size * args.bytes_per_param / 1e6,
                k, shell, shell * args.bytes_per_param / 1e6,
            )
        )
        prev_size = size

    for k in range(len(masks) - 1):
        violations = 0
        for key in masks[k]:
            extra = masks[k][key] & ~masks[k + 1][key]
            violations += int(extra.sum())
        if violations:
            ok = False
            print(
                "NESTING VIOLATION: tier %d touches %d coordinates tier %d does not"
                % (k, violations, k + 1)
            )
        else:
            print("nested: tier %d ⊆ tier %d ✔" % (k, k + 1))

    if not ok:
        print("FAIL: ladder is not nested — shells are ill-defined for this config")
        sys.exit(1)
    print("PASS: ladder is nested; shell table above is valid for TransmissionPlan")


if __name__ == "__main__":
    main()
