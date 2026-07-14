#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=search_pareto_%j.out
#
# dynFedNAS — Pareto search only (skips tiers/population which may already exist).
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_search_pareto.sbatch <CKPT_PATH> <TAG> [--full]
#
#   Default: short budget (max_time_budget=5, population_size=15) — ~30 min, for testing.
#   --full:  production budget (max_time_budget=20, population_size=50) — ~3-4 h.
#
# Output: search_<TAG>_pareto.csv (in the repo root)

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online

CKPT="$1"
TAG="$2"
MODE="${3:-}"

if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
  echo "usage: sbatch cluster/run_search_pareto.sbatch <CKPT_PATH> <TAG> [--full]"; exit 1
fi

CKPT="$(cd "$SLURM_SUBMIT_DIR" && realpath "$CKPT")" || { echo "ckpt not found: $1"; exit 1; }
OUTDIR="$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

if [ "$MODE" = "--full" ]; then
  TIME_BUDGET=500
  POP_SIZE=100
  echo "=== pareto search | tag=$TAG | FULL budget (T=500, pop=100, SuperFedNAS-equivalent) ==="
else
  TIME_BUDGET=5
  POP_SIZE=15
  echo "=== pareto search | tag=$TAG | SHORT budget (T=5, pop=15) — smoke test ==="
fi

python search_phase.py --ckpt "$CKPT" --mode pareto \
  --constraints 0.5 1.0 1.5 2.5 3.5 \
  --bn_recal_batches 0 \
  --fitness_batches 10 \
  --max_time_budget "$TIME_BUDGET" \
  --population_size "$POP_SIZE" \
  --out "$OUTDIR/search_${TAG}_pareto.csv" || exit 1

echo "=== pareto done: search_${TAG}_pareto.csv ==="
