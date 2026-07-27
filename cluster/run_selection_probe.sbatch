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
# WHY STAGE 1 RUNS AT corr=1.0 (decided 2026-07-26, and it is the default here):
# --bw_data_corr couples a client's connectivity to WHICH classes it holds.
#   corr=0   -- poor clients lose deliveries, but they hold the same class mix as
#              everyone else, so only the QUANTITY of data suffers. A selection
#              policy's only lever is reclaiming bytes.
#   corr>0   -- when poor clients fail to deliver, specific classes never reach
#              the model at all (measured in E4: dog and truck at exactly 0.0%).
#              The policy can now recover information available nowhere else.
# Stage 1 is a GATE asking "does the who-participates axis carry any signal?", so
# it is tested where the signal is strongest. No leverage at corr=1.0 means
# certainly none at corr=0, and the retarget dies after 3 runs instead of 6 --
# testing the gate at corr=0 would be testing a drug at its lowest dose.
# Bonus: the inclusion-gap metric is only meaningful at corr>0 (at corr=0 the
# groups hold similar class mixes, so the gap is ~0 BY CONSTRUCTION, not because
# a policy failed) -- see _log_inclusion_gap's docstring.
# AFTER the gate: canonical becomes corr=0.5, with {0, 0.5, 1.0} as a sweep
# figure. corr=1.0 is a declared worst-case bound (DIRECTION §5) and must not
# carry the headline; corr=0 is the control showing the policy does no harm when
# the phenomenon is absent.
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_selection_probe.sbatch <SELECTION> [ROUNDS] [CORR] [DEADLINE] [ALPHA] [JITTER]
#
#   sbatch cluster/run_selection_probe.sbatch none              # blind, the bar
#   sbatch cluster/run_selection_probe.sbatch oracle_bandwidth  # probe A
#   sbatch cluster/run_selection_probe.sbatch coverage          # probe B
#   sbatch cluster/run_selection_probe.sbatch none 3 1.0        # 3-round smoke
#   sbatch cluster/run_selection_probe.sbatch none 75 0.0       # corr=0 control
#   sbatch cluster/run_selection_probe.sbatch none 75 1.0 40 0.5 0.45 nested_blocks
#     # the block stream (§3.1c) with blind selection -- tests whether the
#     # mechanism subsumes probe B's +2.0 WITHOUT its inclusion-gap cost
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
CORR="${3:-1.0}"      # Stage 1 default -- see the note above
DEADLINE="${4:-40}"
ALPHA="${5:-0.5}"
JITTER="${6:-0.45}"
TXMODE="${7:-nested}"   # nested (frozen reference) | nested_blocks (§3.1c)

SEL_FLAG=""
if [ "$SELECTION" != "none" ]; then
  SEL_FLAG="--selection_policy $SELECTION"
fi
RUN_NAME="e6b_${TXMODE}_sel-${SELECTION}_c${CORR}_d${DEADLINE}_a${ALPHA}_j${JITTER}_r${ROUNDS}"
echo "=== dynFedNAS E6b Stage 1 | selection=${SELECTION} corr=${CORR} deadline=${DEADLINE}s alpha=${ALPHA} jitter=${JITTER} rounds=${ROUNDS} ==="

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
  --comm_sim --bw_deadline "$DEADLINE" --bw_seed 0 --tx_mode "$TXMODE" \
  --bw_profile tiered_continuous --bw_jitter_sigma "$JITTER" --bw_data_corr "$CORR" \
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
