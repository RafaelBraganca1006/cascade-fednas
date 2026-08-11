#!/bin/bash
#SBATCH --time=48:00:00
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
#      ... [TXMODE] [MAXNETFIX] [BLOCKORDER] [EVALFREQ] [PIVOT] [SAMPLER] [MODEL] [DARTS_C] [DARTS_LAYERS] [BATCH] [AGGRULE] [SEED]
#
#   sbatch cluster/run_selection_probe.sbatch none              # blind, the bar
#   sbatch cluster/run_selection_probe.sbatch oracle_bandwidth  # probe A
#   sbatch cluster/run_selection_probe.sbatch coverage          # probe B
#   sbatch cluster/run_selection_probe.sbatch none 3 1.0        # 3-round smoke
#   sbatch cluster/run_selection_probe.sbatch none 75 0.0       # corr=0 control
#   sbatch cluster/run_selection_probe.sbatch none 75 1.0 40 0.5 0.45 nested_blocks
#     # the block stream (§3.1c) with blind selection -- tests whether the
#     # mechanism subsumes probe B's +2.0 WITHOUT its inclusion-gap cost
#   sbatch cluster/run_selection_probe.sbatch none 200 0.0 40 0.5 0.45 nested_blocks on fixed 5
#     # the canonical blocks re-run after the classifier fix (86dbe18), eval every
#     # 5 rounds. Replaces e6b_103896, whose numbers carry the bug.
#
# ARGS: <SELECTION> [ROUNDS] [CORR] [DEADLINE] [ALPHA] [JITTER] [TXMODE] [MAXNETFIX]
#       [BLOCKORDER] [EVALFREQ] [STREAM_PIVOT] [SAMPLER] [MODEL] [DARTS_C] [DARTS_LAYERS] [BATCH]
#
#   sbatch cluster/run_selection_probe.sbatch none 200 0.0 40 0.5 0.45 nested_blocks on fixed 10 1
#     # cohort-complementary streaming (DIRECTION 3.1d) vs the shell-major
#     # control at the same config -- decided by [shell_cov] s3
#
# WALLTIME (raised to 24h, 2026-07-29): the `jobs` partition allows 7 days; the
# 12h here was self-imposed. Long-horizon runs need the headroom -- at ~3.3
# min/round for nested_blocks, 200 rounds is ~11h, uncomfortably close to 12h.
# Why 200 rounds matters (DIRECTION §3.3b): our weighted_avg_schedule decays
# MaxNet's beta over num_steps=120, so a 75-round pilot never leaves the warm-up
# phase -- beta is still ~0.36 at round 75 and the max-slot client still takes
# ~36% of the aggregate. 200 rounds clears the decay with 80 rounds to spare.
# It still does NOT match the SuperFedNAS paper's R=1000/1500/2000, so the honest
# phrasing is "after beta's decay completes", never "at full training".
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
TXMODE="${7:-nested}"   # nested (frozen reference) | nested_blocks (§3.1c) | all_or_nothing | none
# MAXNETFIX: on by default, but the VANILLA arm must run without it -- it is our
# addition (DIRECTION §3.3), not SuperFedNAS's. It is provably INERT under
# all_or_nothing anyway (delivery_completeness is only assigned in the nested /
# nested_blocks branches, so the `< 1.0` guard never fires), so switching it off
# there changes no number and removes the question "why does the baseline carry a
# flag you invented?". Same for --bw_estimate, which is only read by the
# capacity_aware / policy_assign samplers and is inert under TS_all_random.
MAXNETFIX="${8:-on}"
BLOCKORDER="${9:-fixed}"   # fixed | random -- within-shell send order (nested_blocks only)
# Eval cadence. Was hardcoded at 25, which gave 9 eval points over a 200-round run
# -- too coarse to support either headline the curve is supposed to carry
# ("reaches the baseline's final accuracy 37% earlier", "3x less variance between
# evals"), and coarse enough to actively mislead: read at round 199 alone the
# three arms look tied, because 199 caught vanilla at a local peak and ours in a
# dip. The converged-phase mean over the last evals says +4.2 / +4.8 points.
# 5 is cheap next to training and makes time-to-accuracy falsifiable.
EVALFREQ="${10:-5}"
# Cohort-complementary streaming (DIRECTION 3.1d). -1 = off, today's shell-major
# stream. 1 = keep the shared core shell-major, complement everything above it:
# clients in the same round start their block-major tail at different blocks, so
# the cohort covers the top shell none of them can upload alone. nested_blocks
# only -- inert in every other tx_mode.
PIVOT="${11:--1}"
# Subnet sampler. TS_all_random (default) is the blind baseline every mechanism
# arm uses. capacity_aware is MaxFit -- assign each client the largest subnet
# that fits its ESTIMATED budget, the obvious fix to the drops. It is the
# thesis's motivating counterexample (best mean, front flattened to +0.09 at
# alpha=100/T=20/uniform), so it has to be re-run at any new profile or the
# headline table has a hole. It reads --bw_estimate, which is why every arm here
# passes `lagged`: oracle would be an unfair advantage for MaxFit specifically.
SAMPLER="${12:-TS_all_random}"
# Model. ofaresnet50_32x32_10_26 is the supernet every DynFedNAS arm uses.
# "darts" is the FedNAS-DARTS baseline (E2c): train.py builds it from
# genotypes.FedNAS_V1 -- the architecture FedNAS published -- with NetworkCIFAR,
# i.e. a FIXED net, no DARTS search. That matches what the anchor did
# (SuperFedNAS 4.1: "the DNN-archs of FedNAS, FedPNAS are kept the same as
# described in that method") and it means the baseline runs on THIS standalone
# path with the same comm_sim, the same 20 clients / 8 per round and the same
# seeds -- no MPI, and none of the memory blowup that killed the distributed
# route (9 DARTS *search* supernets on one GPU: one rank alone took 7.7 GiB).
MODEL="${13:-ofaresnet50_32x32_10_26}"
DARTS_C="${14:-128}"      # channels; sets the payload, see below
DARTS_LAYERS="${15:-20}"
# Local batch size. 64 for every supernet arm. The darts baseline needs less:
# at C=128/layers=20 a single process OOMs at 15.3 GiB with batch 64. Memory is
# dominated by ACTIVATIONS (proportional to batch), the payload by PARAMETERS
# (independent of it) -- so shrinking the batch does not change the payload.
# CORRECTION 2026-08-08: an earlier version of this comment claimed that payload
# was 52.7 MB, "between our S (41.6 MB) and M (98.6 MB) tiers". That is WRONG.
# NetworkCIFAR(C=128, layers=20) is 6.61 M parameters = 201.9 MB, i.e. between
# our L (137.6 MB) and XL (286.9 MB) tiers -- measured directly in
# analysis/sweep_deadline.py. The wrong figure was never load-bearing for a
# claim, but it did travel into a figure caption before being caught, and 202 MB
# is what actually explains the arm's 95% drop rate (44.7 MB gives ~49%).
# The arm's claim is comm_sim ON vs OFF, internal to FedNAS, so the batch
# difference does not touch it; state it in the text.
BATCH="${16:-64}"

# Aggregation rule (DIRECTION 3.3c). `auto` reproduces the historical behaviour
# exactly -- completeness scaling when MAXNETFIX=on, the inherited slot weight
# otherwise -- so every archived invocation is unaffected. `cascade` weights each
# emitted part by the inverse frequency with which THAT client reaches THAT
# region, and the trainer refuses it on any mode other than nested_blocks,
# because only the block stream emits parts that map to regions.
AGGRULE="${17:-auto}"

# Seed. Sets BOTH the weight init and the bandwidth chain, so a seed is a whole
# alternative world. Arms compared within one seed therefore share a bandwidth
# trace, which is what makes them paired -- E1's rule, "same bandwidth seeds
# across methods".
#
# Worth knowing before reading small differences: cudnn determinism is disabled
# on this path (train.py:522 is commented out), so two runs with the SAME seed
# still diverge. Measured on an accidental replicate pair: 0.64 on mean accuracy
# and 1.44 on spread by round 149. Nothing below that is a result without seeds.
SEED="${18:-0}"

# The DARTS baseline has ONE architecture, so the supernet machinery has to be
# switched off for it: TS_all_random calls get_subnet(**arch_config) and
# ServerDarts returns a Genotype, not a mapping (that is the crash), and
# MaxNet's beta schedule is meaningless with a single subnet -- FedNAS is plain
# FedAvg over one net. `static` + a single empty tier reproduces exactly that.
SUBNETS='{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}'
AVGSCHED='{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}'
SCHED_FLAG="--weighted_avg_schedule $AVGSCHED"
if [ "$MODEL" = "darts" ]; then
  SUBNETS='{"0":{}}'
  SAMPLER="static"
  SCHED_FLAG=""
  echo "=== darts: single-architecture baseline -> sampler=static, no MaxNet schedule ==="
fi

SEL_FLAG=""
if [ "$SELECTION" != "none" ]; then
  SEL_FLAG="--selection_policy $SELECTION"
fi
# tx_mode "none" = the no-communication ceiling: --comm_sim omitted entirely, so
# nothing is ever dropped or truncated. It is the upper bound the mechanism is
# measured against ("dynamic bandwidth costs vanilla X points; we recover Y%").
if [ "$TXMODE" = "none" ]; then
  COMM_FLAGS=""
else
  COMM_FLAGS="--comm_sim --bw_deadline $DEADLINE --bw_seed $SEED --tx_mode $TXMODE"
  COMM_FLAGS="$COMM_FLAGS --bw_profile tiered_continuous --bw_jitter_sigma $JITTER"
  COMM_FLAGS="$COMM_FLAGS --bw_data_corr $CORR --bw_estimate lagged"
  if [ "$MAXNETFIX" = "on" ]; then
    COMM_FLAGS="$COMM_FLAGS --maxnet_delivery_aware"
  fi
fi
RUN_NAME="e6b_${MODEL:0:6}_${TXMODE}_samp${SAMPLER}_ord${BLOCKORDER}_piv${PIVOT}_mx${MAXNETFIX}_agg${AGGRULE}_sel-${SELECTION}_c${CORR}_d${DEADLINE}_a${ALPHA}_j${JITTER}_r${ROUNDS}_s${SEED}"
echo "=== dynFedNAS E6b Stage 1 | selection=${SELECTION} corr=${CORR} deadline=${DEADLINE}s alpha=${ALPHA} jitter=${JITTER} rounds=${ROUNDS} ==="

cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"
python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model "$MODEL" --init_channel_size "$DARTS_C" --darts_layers "$DARTS_LAYERS" \
  --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round "$ROUNDS" \
  --epochs 5 --batch_size "$BATCH" --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type "$SAMPLER" \
  --diverse_subnets "$SUBNETS" \
  $SCHED_FLAG \
  --frequency_of_the_test "$EVALFREQ" --partition_alpha "$ALPHA" --init_seed "$SEED" \
  --aggregation_rule "$AGGRULE" \
  --efficient_test --verbose --log_per_class --log_inclusion_gap \
  --block_order "$BLOCKORDER" --stream_pivot "$PIVOT" \
  $COMM_FLAGS \
  $SEL_FLAG \
  --wandb_project_name dynfednas --wandb_run_name "$RUN_NAME" --wandb_entity rbraganca-eth-z-rich

OUT="$SLURM_SUBMIT_DIR/e6b_$SLURM_JOB_ID.out"
echo "=== E6b done: $RUN_NAME -- summary ==="
grep "connectivity tiers" "$OUT" 2>/dev/null | head -1
grep "bandwidth spread"   "$OUT" 2>/dev/null | head -1
grep "\[selection\] round" "$OUT" 2>/dev/null | tail -1
grep "\[inclusion\]"       "$OUT" 2>/dev/null | tail -4
grep "run summary"         "$OUT" 2>/dev/null | tail -1
