#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=v2_arm_%j.out
#
# dynFedNAS v2 — one arm of the mechanism experiment (IMPLEMENTATION_PLAN
# Stage 4): vanilla TS sampling under a chosen deadline and transmission mode.
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_v2_arm.sbatch <DEADLINE_S> <TX_MODE> [ROUNDS] [SUBNET_DIST_TYPE] [ABLATION] [ASSIGNMENT_POLICY]
#
#   sbatch cluster/run_v2_arm.sbatch 20 nested_shard 20    # shard smoke
#   sbatch cluster/run_v2_arm.sbatch 20 all_or_nothing 150 # three-arm harsh (a)
#   sbatch cluster/run_v2_arm.sbatch 20 nested 150         # three-arm harsh (b)
#   sbatch cluster/run_v2_arm.sbatch 20 nested_shard 150   # three-arm harsh (c)
#   sbatch cluster/run_v2_arm.sbatch 20 all_or_nothing 150 static  # E2b: alt sampler
#   sbatch cluster/run_v2_arm.sbatch 20 nested 150 TS_all_random ablation  # E3
#   sbatch cluster/run_v2_arm.sbatch 20 nested 150 policy_assign none greedy  # E5 greedy
#
# Same model/subnets/schedule/seeds as run_sweep / run_phase3 -> directly
# comparable across arms and with the v1 characterization runs (when
# SUBNET_DIST_TYPE is left at its default, TS_all_random). The E3 ablation run
# above is DIRECTLY comparable to the existing arm (b) baseline
# (results/v2_arms/v2_arm_harsh_nested.out, same deadline/tx_mode/rounds/seeds)
# without needing to rerun the baseline -- nothing in the aggregation/
# truncation path changed since that run. Same for the E5 greedy run above:
# SUBNET_DIST_TYPE=policy_assign requires ASSIGNMENT_POLICY (5th arg is the
# ablation slot, so pass "none" there to skip it and put the policy 6th).

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online   # requires `wandb login` once on the cluster

DEADLINE="$1"
TX_MODE="$2"
ROUNDS="${3:-150}"
SUBNET_DIST="${4:-TS_all_random}"
ABLATION_FLAG=""
RUN_SUFFIX=""
if [ "$5" = "ablation" ]; then
  ABLATION_FLAG="--nested_context_ablation"
  RUN_SUFFIX="_ctxablation"
fi
POLICY_FLAG=""
if [ -n "$6" ]; then
  if [ "$SUBNET_DIST" != "policy_assign" ]; then
    echo "ASSIGNMENT_POLICY given but SUBNET_DIST_TYPE is not policy_assign"; exit 1
  fi
  POLICY_FLAG="--assignment_policy $6"
  RUN_SUFFIX="${RUN_SUFFIX}_policy${6}"
fi
if [ -z "$DEADLINE" ] || [ -z "$TX_MODE" ]; then
  echo "usage: sbatch cluster/run_v2_arm.sbatch <DEADLINE_S> <TX_MODE> [ROUNDS] [SUBNET_DIST_TYPE] [ABLATION] [ASSIGNMENT_POLICY]"; exit 1
fi
RUN_NAME="v2_${TX_MODE}_d${DEADLINE}_r${ROUNDS}_${SUBNET_DIST}${RUN_SUFFIX}"
echo "=== dynFedNAS v2 arm | deadline=${DEADLINE}s tx_mode=${TX_MODE} rounds=${ROUNDS} subnet_dist=${SUBNET_DIST} ablation=${ABLATION_FLAG:-off} policy=${6:-off} ==="

cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"
python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round "$ROUNDS" \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type "$SUBNET_DIST" \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha 100 --init_seed 0 \
  --efficient_test --verbose \
  --comm_sim --bw_deadline "$DEADLINE" --bw_seed 0 --tx_mode "$TX_MODE" \
  --shard_carriers 2 $ABLATION_FLAG $POLICY_FLAG \
  --wandb_project_name dynfednas --wandb_run_name "$RUN_NAME" --wandb_entity rbraganca-eth-z-rich

OUT="$SLURM_SUBMIT_DIR/v2_arm_$SLURM_JOB_ID.out"
echo "=== arm done: $RUN_NAME — summary ==="
echo "truncated deliveries: $(grep -c 'truncated: trained' "$OUT" 2>/dev/null)"
echo "carrier deliveries:   $(grep -c 'carrier .* delivered' "$OUT" 2>/dev/null)"
grep "run summary" "$OUT" 2>/dev/null | tail -1
