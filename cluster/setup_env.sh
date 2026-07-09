#!/bin/bash
# dynfednas — one-time environment setup on the D-INFK student cluster.
#
# Run on a COMPUTE node (so the torch CUDA build matches the GPU), e.g.:
#   srun --pty -A projects -t 60 bash --login
#   bash cluster/setup_env.sh
set -e

. /etc/profile.d/modules.sh
module add cuda/12.9

VENV=/work/scratch/$USER/dynfednas-venv     # keep venv in scratch (home is only 20 GB)
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# 1) modern torch/torchvision (CUDA 12.x build)
pip install torch torchvision

# 2) public OFA package WITHOUT deps, so it does not drag in ancient torch
pip install --no-deps ofa==0.1.0.post202012082159

# 3) remaining runtime deps (CIFAR-10 path — no tensorflow needed)
pip install numpy scipy matplotlib tqdm scikit-learn networkx PyYAML Pillow h5py wandb

# 4) verify every OFA symbol SuperFedNAS imports resolves (torch._six shim included)
python - <<'PY'
import sys, types, queue, collections.abc, torch
if not hasattr(torch, "_six"):
    m = types.ModuleType("torch._six"); m.queue = queue
    m.string_classes = (str,); m.int_classes = (int,); m.container_abcs = collections.abc
    sys.modules["torch._six"] = m
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d, DynamicLinear
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import copy_bn
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.imagenet_classification.networks import ResNets, MobileNetV3
from ofa.utils import val2list, make_divisible, SEModule, MyNetwork, MyGlobalAvgPool2d
from ofa.utils.layers import IdentityLayer, ResidualBlock, ResNetBottleneckBlock, LinearLayer
from ofa.utils import flops_counter
from ofa.utils.pytorch_utils import get_net_device
print("ALL OFA IMPORTS OK — cuda available:", torch.cuda.is_available())
PY

echo "Setup complete. venv at: $VENV"
