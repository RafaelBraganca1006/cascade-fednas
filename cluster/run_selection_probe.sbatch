#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=e6b_%j.out
#
# dynFedNAS -- E6b Stage 1: client-SELECTION leverage probes.
# Design: DIRECTION.md §3.2b. Plan + gate: EXPERIMENTS.md E6b.
#
# The question this batch answers is NOT "which selector is best". It is
# "does the who-participates axis carry any signal at all?", asked BEFORE
# three weeks are spent building an RL on top of it. The two probes are
# single-signal extremes, not ceilings: oracle_bandwidth can land BELOW blind
# under non-IID (always picking the fastest clients is the digital-divide
# pathology this thesis argues against), and that is an informative result.
#
#   Gate -- both probes within ~0.05 mean accuracy of blind => the axis is dead,
#   stop and take the negative-result framing. Either probe departing clearly
#   from blind, IN EITHER DIRECTION => leverage exists, Stage 2 is justified.
#   What kills the retarget is a TIE, not probe A losing.
#
# Canonical config chosen offline by analysis/pick_deadline.py (S0.1):
# tiered_continuous / jitter 0.45 / T=40 puts every tier in reach (S 14.7% /
# M 8.1% / L 27.3% / XL 15.1% of client-rounds) while leaving ~35% of
# participation slots structurally dead -- live action space AND a live lever.
# T=20 had L and XL unreachable by anyone in any round, which is what made the
# E5/E6 assignment comparison degenerate.
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_selection_probe.sbatch none              # blind, the bar
#   sbatch cluster/run_selection_probe.sbatch oracle_bandwidth  # probe A
#   sbatch cluster/run_selection_probe.sbatch coverage          # probe B
#   sbatch cluster/run_selection_probe.sbatch none 150          # full length
#
# Note --bw_estimate lagged: every E5/E6 policy run silently used the parser
# default (oracle) and still lost to blind (DIRECTION §3.2b fact 5). The probes
# read oracle bandwidth internally by design -- they are diagnostics -- but
# everything else in the pipeline now sees the realistic stale estimate.

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online

SELECTION="${1:-none}"
ROUNDS="${2:-75}"
DEADLINE="${3:-40}"
ALPHA="${4:-0.5}"
JITTER="${5:-0.45}"

SEL_FLAG=""
if [ "$SELECTION" != "none" ]; then
  SEL_FLAG="--selection_policy $SELECTION"
fi
RUN_NAME="e6b_sel-${SELECTION}_d${DEADLINE}_a${ALPHA}_j${JITTER}_r${ROUNDS}"
echo "=== dynFedNAS E6b Stage 1 | selection=${SELECTION} deadline=${DEADLINE}s alpha=${ALPHA} jitter=${JITTER} rounds=${ROUNDS} ==="

cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"
python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round "$ROUNDS" \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type TS_all_random \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha "$ALPHA" --init_seed 0 \
  --efficient_test --verbose --log_per_class --log_inclusion_gap \
  --comm_sim --bw_deadline "$DEADLINE" --bw_seed 0 --tx_mode nested \
  --bw_profile tiered_continuous --bw_jitter_sigma "$JITTER" --bw_data_corr 0.0 \
  --bw_estimate lagged --maxnet_delivery_aware \
  $SEL_FLAG \
  --wandb_project_name dynfednas --wandb_run_name "$RUN_NAME" --wandb_entity rbraganca-eth-z-rich

OUT="$SLURM_SUBMIT_DIR/e6b_$SLURM_JOB_ID.out"
echo "=== E6b done: $RUN_NAME -- summary ==="
grep "connectivity tiers" "$OUT" 2>/dev/null | head -1
grep "bandwidth spread"   "$OUT" 2>/dev/null | head -1
grep "\[selection\] round" "$OUT" 2>/dev/null | tail -1
grep "\[inclusion\]"       "$OUT" 2>/dev/null | tail -4
grep "run summary"         "$OUT" 2>/dev/null | tail -1
