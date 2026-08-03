#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=shape_%j.out
#
# dynFedNAS -- the SHAPE-ALIGNMENT check (DIRECTION 3.3b-old, TODO backlog #1).
#
# THE RISK IT TESTS, and it is the only open one that can still invalidate the
# accuracy axis. Under the truncating modes a delivered update is uniform
# e=0.1 -- which is EXACTLY the shape of the four evaluation probes
# ({"d":[0,0,0,0],"e":0.1} and its siblings). So the mechanism arms might score
# higher not because they train the supernet better, but because what they
# deliver happens to match what we measure. Every accuracy number in the
# canonical table rests on those four probes.
#
# THE TEST. Evaluate each checkpoint twice:
#   --mode tiers       the four probes (reproduces the training-log numbers)
#   --mode population  100 RANDOM architectures from the elastic space
# If the gap between arms survives on the random population, the advantage is
# real. If it collapses, it was shape alignment and the accuracy axis has to be
# rewritten around the population numbers instead.
#
# --mode pareto is deliberately NOT run here: it is the expensive one (a --full
# run once hit the 12h walltime at 77/500 generations) and it answers a
# different question.
#
# NOT zero-GPU, despite how the backlog first described it: no TRAINING happens,
# but evaluating 100 architectures is inference on the full test set and needs a
# card and a queue slot. Budget ~30-45 min per checkpoint per mode.
#
# USAGE -- pass CKPT:TAG pairs, any number:
#   sbatch cluster/run_shape_alignment.sbatch \
#     fedml_experiments/standalone/superfednas/wandb/run-.../files/best_checkpoint_supernet_1000.pt:aon \
#     fedml_experiments/standalone/superfednas/wandb/run-.../files/best_checkpoint_supernet_1000.pt:blocks

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=offline    # nothing here is a training run worth tracking

if [ "$#" -lt 1 ]; then
  echo "usage: sbatch cluster/run_shape_alignment.sbatch <CKPT:TAG> [<CKPT:TAG> ...]" >&2
  exit 2
fi

OUTDIR="$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

for pair in "$@"; do
  CKPT="${pair%:*}"
  TAG="${pair##*:}"
  CKPT="$(cd "$SLURM_SUBMIT_DIR" && realpath "$CKPT")" || { echo "ckpt not found: $pair" >&2; exit 1; }
  echo "=== shape-alignment | tag=$TAG ==="
  echo "    ckpt=$CKPT"

  python -u search_phase.py --ckpt "$CKPT" --mode tiers \
    --bn_recal_batches 0 \
    --out "$OUTDIR/shape_${TAG}_tiers.csv" || exit 1

  python -u search_phase.py --ckpt "$CKPT" --mode population --pop 100 \
    --bn_recal_batches 0 \
    --out "$OUTDIR/shape_${TAG}_population.csv" || exit 1

  echo "=== $TAG done ==="
done

echo "=== all checkpoints done ==="
ls -la "$OUTDIR"/shape_*_{tiers,population}.csv 2>/dev/null
