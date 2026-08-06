#!/bin/bash
#SBATCH --job-name=fedavg_commsim
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=fedavg_%j.out
#
# dynFedNAS -- FedAvg under the round deadline (docs/EXPERIMENTS.md E10).
#
# WHAT THIS IS, AND WHY IT IS NOT AN ADAPTATION. This runs FedML's own
# standalone FedAvg (fedml_api/standalone/fedavg/), untouched: its aggregation,
# its client sampling, its local training, its models. The single intervention
# is that a client whose update misses the round deadline is not appended to
# w_locals. Nothing about FedAvg is reinterpreted or reimplemented -- we add the
# channel, which is the object under study and which no FL codebase models.
#
# This replaces the FedNAS-DARTS arm (E2c), cut 2026-08-06 after job 104233:
# that arm scored exactly 10.0% (chance) in BOTH the comm_sim ON and OFF
# conditions, i.e. the model never trained, so the ON number carried no
# information about bandwidth. 43.9 h for nothing. The lesson is written into
# the run rule below.
#
# RUN RULE, NON-NEGOTIABLE: smoke first (ROUNDS=3) and read the training loss.
# If it is not clearly below ln(10) = 2.303 and falling, STOP -- the arm is not
# learning and a 150-round run will only prove it more expensively. This is
# exactly the check that would have saved job 104233.
#
# WHY FedAvg IS LOW RISK HERE, verified in the code before writing this:
#   1. main_fedavg.py supports cifar10 natively with resnet56 as its default,
#      and imports the SAME load_partition_data_cifar10 our trainer uses -- same
#      partition, same --partition_alpha.
#   2. _client_sampling uses np.random.seed(round_idx), identical to the
#      superfednas trainer, so client selection is seed-matched for free.
#   3. _aggregate normalises by the sum of sample counts over w_locals, so a
#      dropped client renormalises correctly with no change. The trap that had
#      to be fixed by hand in FedNASAggregator (dividing by the total over ALL
#      clients, silently scaling the aggregate down every round) does not exist.
#   4. resnet56 is a standard fixed model, not a 20-layer DARTS with a hardcoded
#      drop_path_prob of 0.5 -- the cause of the 104233 failure has no analogue.
#
# WHAT THIS ARM CAN AND CANNOT SUPPORT (settled 2026-08-06):
#   YES -- drop / waste / attempted bytes; accuracy with comm_sim ON vs OFF,
#          which is internal to the arm; and the O(N) -> O(1) communication
#          argument (serving N deployment targets costs N full FL runs of uplink
#          with FedAvg and one with the supernet).
#   NO  -- spread. One fixed architecture has no Pareto front to lose. That
#          absence IS the finding, not a limitation of the experiment, and it is
#          the measured half of the impossibility argument in docs/THESIS.md 2b:
#          a fixed model has no subset structure, so a prefix of its update is
#          not a smaller valid model. FedAvg CANNOT have anytime delivery.
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_fedavg_commsim.sbatch <COMM> [ROUNDS] [ALPHA] [DEADLINE] [MODEL] [BATCH] [EVALFREQ]
#
#   sbatch cluster/run_fedavg_commsim.sbatch on  3          # THE SMOKE -- run this first
#   sbatch cluster/run_fedavg_commsim.sbatch on  150        # ON  at the canonical config
#   sbatch cluster/run_fedavg_commsim.sbatch off 150        # OFF, the control that makes ON mean something
#
# COMM=off omits --comm_sim entirely. Note this arm is NOT seed-comparable with
# COMM=on in the strict sense (the tiered path walks every train loader once to
# build the label histograms and consumes torch RNG -- the same asymmetry that
# made e6b_103804 unusable). It does not matter here: the claim is ON vs OFF at
# the level of "does the deadline destroy this method", not a paired per-round
# comparison. Say so in the text rather than implying parity.

set -euo pipefail

COMM="${1:-on}"
ROUNDS="${2:-150}"
ALPHA="${3:-100}"
DEADLINE="${4:-20}"
MODEL="${5:-resnet56}"
BATCH="${6:-64}"
EVALFREQ="${7:-10}"

COMM_FLAGS=""
if [ "$COMM" = "on" ]; then
  # tiered_continuous / jitter 0.45 / corr 0 / seed 0 -- the canonical config,
  # identical to every supernet arm so the byte axis is comparable.
  COMM_FLAGS="--comm_sim --bw_deadline $DEADLINE --bw_seed 0"
  COMM_FLAGS="$COMM_FLAGS --bw_profile tiered_continuous --bw_jitter_sigma 0.45"
  COMM_FLAGS="$COMM_FLAGS --bw_data_corr 0.0"
fi

echo "=== FedAvg + comm_sim: comm=$COMM rounds=$ROUNDS alpha=$ALPHA T=$DEADLINE model=$MODEL batch=$BATCH ==="
nvidia-smi || true

source /work/scratch/$USER/dynfednas-venv/bin/activate
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/fedavg"

# 20 clients / 8 per round / 5 local epochs / lr 0.1 / sgd: matched to every
# supernet arm so the only differences are the method and the model.
python main_fedavg.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model "$MODEL" \
  --partition_method hetero --partition_alpha "$ALPHA" \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round "$ROUNDS" \
  --epochs 5 --batch_size "$BATCH" --client_optimizer sgd --lr 0.1 --ci 0 \
  --frequency_of_the_test "$EVALFREQ" \
  $COMM_FLAGS

OUT="$SLURM_SUBMIT_DIR/fedavg_$SLURM_JOB_ID.out"
echo "=== FedAvg done -- summary ==="
grep "\[comm_sim\] enabled"  "$OUT" 2>/dev/null | head -1
grep "\[comm_sim\] round"    "$OUT" 2>/dev/null | tail -1
echo "--- SMOKE CHECK: the loss must be clearly below ln(10)=2.303 and falling."
grep -oE "Loss: [0-9.]+" "$OUT" 2>/dev/null | head -3
grep -oE "Loss: [0-9.]+" "$OUT" 2>/dev/null | tail -3
