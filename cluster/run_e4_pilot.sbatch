#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=e4_pilot_%j.out
#
# dynFedNAS — E4 pilot: non-IID (Dirichlet) + heterogeneous connectivity
# (tiered profiles) + optional bandwidth-data correlation (EXPERIMENTS.md E4).
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_e4_pilot.sbatch <DEADLINE_S> <TX_MODE> <ALPHA> <CORR> [ROUNDS] [SUBNET_DIST_TYPE] [ASSIGNMENT_POLICY] [MAXNET_FIX]
#
#   sbatch cluster/run_e4_pilot.sbatch 20 all_or_nothing 0.5 0.0 75   # non-IID alone
#   sbatch cluster/run_e4_pilot.sbatch 20 nested         0.5 0.0 75
#   sbatch cluster/run_e4_pilot.sbatch 20 nested         0.5 1.0 75   # + digital divide
#   sbatch cluster/run_e4_pilot.sbatch 20 nested 0.5 1.0 75 policy_assign greedy maxnetfix
#     # does greedy protect the corr=1.0 blind-spot classes any better than
#     # blind TS_all_random? (it's bandwidth-blind too -- expect: probably not)
#
# Differences vs run_v2_arm.sbatch (kept identical otherwise, for attribution):
#   --partition_alpha ALPHA   (v2 arms used 100 = near-IID)
#   --bw_profile tiered       persistent poor/average/good connectivity
#   --bw_data_corr CORR       0 = independent, 1 = poor clients hold low classes
#   --log_per_class           per-class acc at eval checkpoints (heatmap data)
#
# Pilots (EXPERIMENTS.md two-phase strategy): 75 rounds, single seed —
# design decisions only, numbers never enter the thesis text. corr=1.0 is a
# worst-case sensitivity bound, not a real-world prevalence claim (DIRECTION §5).

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online

DEADLINE="$1"
TX_MODE="$2"
ALPHA="$3"
CORR="$4"
ROUNDS="${5:-75}"
SUBNET_DIST="${6:-TS_all_random}"
POLICY_FLAG=""
RUN_SUFFIX=""
if [ -n "$7" ] && [ "$7" != "none" ]; then
  if [ "$SUBNET_DIST" != "policy_assign" ]; then
    echo "ASSIGNMENT_POLICY given but SUBNET_DIST_TYPE is not policy_assign"; exit 1
  fi
  POLICY_FLAG="--assignment_policy $7"
  RUN_SUFFIX="_policy${7}"
fi
MAXNETFIX_FLAG=""
if [ "$8" = "maxnetfix" ]; then
  MAXNETFIX_FLAG="--maxnet_delivery_aware"
  RUN_SUFFIX="${RUN_SUFFIX}_maxnetfix"
fi
if [ -z "$DEADLINE" ] || [ -z "$TX_MODE" ] || [ -z "$ALPHA" ] || [ -z "$CORR" ]; then
  echo "usage: sbatch cluster/run_e4_pilot.sbatch <DEADLINE_S> <TX_MODE> <ALPHA> <CORR> [ROUNDS] [SUBNET_DIST_TYPE] [ASSIGNMENT_POLICY] [MAXNET_FIX]"; exit 1
fi
RUN_NAME="e4_${TX_MODE}_d${DEADLINE}_a${ALPHA}_c${CORR}_r${ROUNDS}_${SUBNET_DIST}${RUN_SUFFIX}"
echo "=== dynFedNAS E4 pilot | deadline=${DEADLINE}s tx=${TX_MODE} alpha=${ALPHA} corr=${CORR} rounds=${ROUNDS} subnet_dist=${SUBNET_DIST} policy=${7:-off} maxnetfix=${MAXNETFIX_FLAG:-off} ==="

cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"
python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round "$ROUNDS" \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type "$SUBNET_DIST" \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha "$ALPHA" --init_seed 0 \
  --efficient_test --verbose --log_per_class \
  --comm_sim --bw_deadline "$DEADLINE" --bw_seed 0 --tx_mode "$TX_MODE" \
  --bw_profile tiered --bw_data_corr "$CORR" \
  --shard_carriers 2 $POLICY_FLAG $MAXNETFIX_FLAG \
  --wandb_project_name dynfednas --wandb_run_name "$RUN_NAME" --wandb_entity rbraganca-eth-z-rich

OUT="$SLURM_SUBMIT_DIR/e4_pilot_$SLURM_JOB_ID.out"
echo "=== E4 pilot done: $RUN_NAME — summary ==="
grep "connectivity tiers" "$OUT" 2>/dev/null | head -1
grep "run summary" "$OUT" 2>/dev/null | tail -1
