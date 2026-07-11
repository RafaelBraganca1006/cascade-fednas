#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=v2_stage0_%j.out
#
# dynFedNAS v2 — Stage 0 acceptance checks (IMPLEMENTATION_PLAN.md).
#
#   1. verify_nesting: the configured tier ladder is nested in supernet
#      coordinates (gates the whole shell design — Stage 1.1).
#   2. Legacy smoke: 20 rounds, vanilla TS_all_random + comm_sim in
#      all_or_nothing mode. The per-round "[comm_sim] round" lines must be
#      byte-identical to v1 (the round_time fix only changes wandb
#      round_time_s/wall_clock metrics, not these lines).
#
# Usage (submit from the dynfednas_v2 repo root):
#   sbatch cluster/run_v2_stage0_checks.sbatch                    # checks on v2 only
#   sbatch cluster/run_v2_stage0_checks.sbatch /path/to/dynfednas # + auto-diff vs v1
#
# With a v1 repo path as $1, the same smoke runs there and the comm lines are
# diffed automatically: expect "IDENTICAL" at the end.

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=offline

V2_ROOT="$SLURM_SUBMIT_DIR"
V1_ROOT="$1"

SMOKE_ARGS=(
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10
  --model ofaresnet50_32x32_10_26 --partition_method hetero
  --client_num_in_total 20 --client_num_per_round 8 --comm_round 20
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0
  --subnet_dist_type TS_all_random
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}'
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}'
  --frequency_of_the_test 25 --partition_alpha 100 --init_seed 0
  --efficient_test --verbose
  --comm_sim --bw_deadline 60 --bw_seed 0
  --wandb_project_name dynfednas --wandb_run_name v2_stage0_smoke --wandb_entity none
)

echo "=== check 1: tier-ladder nestedness (verify_nesting) ==="
cd "$V2_ROOT"
python fedml_api/standalone/superfednas/comm_sim/verify_nesting.py || exit 1

echo "=== check 2: v2 legacy smoke (20 rounds, all_or_nothing) ==="
cd "$V2_ROOT/fedml_experiments/standalone/superfednas"
python train.py "${SMOKE_ARGS[@]}" 2>&1 | tee /tmp/v2_smoke_$SLURM_JOB_ID.log
grep "\[comm_sim\] round" /tmp/v2_smoke_$SLURM_JOB_ID.log > /tmp/v2_comm_$SLURM_JOB_ID.txt
echo "--- v2 comm lines: $(wc -l < /tmp/v2_comm_$SLURM_JOB_ID.txt) rounds ---"

if [ -n "$V1_ROOT" ]; then
  echo "=== check 3: v1 reference smoke + diff ==="
  cd "$V1_ROOT/fedml_experiments/standalone/superfednas"
  python train.py "${SMOKE_ARGS[@]}" 2>&1 | tee /tmp/v1_smoke_$SLURM_JOB_ID.log
  grep "\[comm_sim\] round" /tmp/v1_smoke_$SLURM_JOB_ID.log > /tmp/v1_comm_$SLURM_JOB_ID.txt
  if diff /tmp/v1_comm_$SLURM_JOB_ID.txt /tmp/v2_comm_$SLURM_JOB_ID.txt; then
    echo "STAGE 0 REGRESSION: IDENTICAL — v2 legacy mode reproduces v1"
  else
    echo "STAGE 0 REGRESSION: MISMATCH — investigate before Stage 1"; exit 1
  fi
else
  echo "(no v1 repo path given — run with the v1 path to auto-diff, or compare"
  echo " /tmp/v2_comm_$SLURM_JOB_ID.txt against a v1 20-round run manually)"
fi
echo "=== stage 0 checks done ==="
