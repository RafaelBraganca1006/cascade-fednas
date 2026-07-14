#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=search_%j.out
#
# dynFedNAS — Search Phase over a trained supernet checkpoint: tier sanity,
# random-population cloud, and the evolutionary Pareto front (DIRECTION §5).
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_search_phase.sbatch <CKPT_PATH> <TAG>
#
#   <CKPT_PATH>: .../wandb/offline-run-*/files/finished_checkpoint_data.pt
#   <TAG>:       label for output CSVs, e.g. harsh_nested_shard
#
# Find checkpoints with:
#   ls fedml_experiments/standalone/superfednas/wandb/offline-run-*/files/finished_checkpoint_data.pt
# (match run dir to arm via the wandb run name in that dir's files/config.yaml)
#
# Outputs (in the repo root): search_<TAG>_{tiers,population,pareto}.csv

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online   # requires `wandb login` once on the cluster

CKPT="$1"
TAG="$2"
if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
  echo "usage: sbatch cluster/run_search_phase.sbatch <CKPT_PATH> <TAG>"; exit 1
fi
# resolve to absolute BEFORE cd'ing into the experiment dir (relative paths
# from the submit dir would otherwise break torch.load)
CKPT="$(cd "$SLURM_SUBMIT_DIR" && realpath "$CKPT")" || { echo "ckpt not found: $1"; exit 1; }
OUTDIR="$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

echo "=== search phase | ckpt=$CKPT tag=$TAG ==="
python -u search_phase.py --ckpt "$CKPT" --mode tiers \
  --bn_recal_batches 0 \
  --out "$OUTDIR/search_${TAG}_tiers.csv" || exit 1
python -u search_phase.py --ckpt "$CKPT" --mode population --pop 100 \
  --bn_recal_batches 0 \
  --out "$OUTDIR/search_${TAG}_population.csv" || exit 1
python -u search_phase.py --ckpt "$CKPT" --mode pareto \
  --constraints 0.2 0.4 0.8 1.6 3.2 \
  --bn_recal_batches 0 \
  --out "$OUTDIR/search_${TAG}_pareto.csv" || exit 1
echo "=== search phase done: search_${TAG}_{tiers,population,pareto}.csv ==="
