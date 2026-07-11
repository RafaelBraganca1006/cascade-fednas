#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=v2_nested_smoke_%j.out
#
# dynFedNAS v2 — Stage 1 acceptance: first end-to-end run of nested anytime
# transmission (IMPLEMENTATION_PLAN.md Stage 1.2-1.5).
#
# 20 rounds, vanilla TS_all_random, HARSH deadline (T=20s) in nested mode —
# harsh is where truncation actually fires (legacy drops 75% there).
#
# What to look for in the .out:
#   - "[comm_sim] client N truncated: trained {...}, delivered tier K {...}"
#     lines: the anytime mechanism working (train α, deliver β ⊂ α).
#   - participation up vs legacy harsh (~0.25): most former droppers now
#     deliver a smaller tier instead of nothing.
#   - comm/waste_rate near 0 (only cut-shell remainders).
#   - no crash in _slice_client_update (the new torch path).
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_v2_nested_smoke.sbatch

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=offline
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round 20 \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type TS_all_random \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha 100 --init_seed 0 \
  --efficient_test --verbose \
  --comm_sim --bw_deadline 20 --bw_seed 0 --tx_mode nested \
  --wandb_project_name dynfednas --wandb_run_name v2_nested_smoke_harsh \
  --wandb_entity none

OUT="$SLURM_SUBMIT_DIR/v2_nested_smoke_$SLURM_JOB_ID.out"
echo "=== nested smoke done — summary ==="
echo "truncated deliveries: $(grep -c 'truncated: trained' "$OUT" 2>/dev/null)"
grep "\[comm_sim\] round" "$OUT" 2>/dev/null | tail -5 || true
