#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=baseline_smoke_%j.out
#
# dynfednas — Phase 0 smoke test: plain FedAvg on ONE fixed subnet, 50 rounds.
# Confirms the environment runs end-to-end and accuracy rises before any modification.
# Submit from the repo root:  sbatch cluster/run_baseline_smoke.sbatch
# Prerequisite: CIFAR-10 downloaded + split under data/cifar10 (README steps 3-4).

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate

export WANDB_MODE=offline
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round 50 \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type static --diverse_subnets '{"0":{"d":[2,2,2,2],"e":0.25}}' \
  --frequency_of_the_test 10 --partition_alpha 100 --init_seed 0 \
  --efficient_test --verbose \
  --wandb_project_name dynfednas --wandb_run_name smoke --wandb_entity none
