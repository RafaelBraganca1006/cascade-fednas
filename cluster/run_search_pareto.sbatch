#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=search_pareto_%j.out
#
# WHY THE LOWEST CONSTRAINT IS 1.0 AND NOT LOWER (measured 2026-08-20).
# A 0.5 GFLOP ceiling was tried to cover the smallest ladder tier and job 106277
# timed out after 4 h having admitted ONE architecture from 892 random samples.
# The minimum subnet is 0.459 GFLOPs, so a 0.5 ceiling only accepts architectures
# sitting on the floor of the space, which random sampling essentially never
# produces; a population of 50 would need ~45k draws. The tier is also the
# minimum itself, so there is nothing there to search for. Constraints must stay
# where random sampling has room above the floor.
#
# dynFedNAS — Pareto search only (skips tiers/population which may already exist).
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_search_pareto.sbatch <CKPT_PATH> <TAG> [--full]
#
#   Default: short budget (max_time_budget=5, population_size=15) — ~30 min, for testing.
#   --full:  T=25, pop=50 — ~1.5-2h (fixed 2026-07-22: this used to say T=20/pop=50
#            in this comment but the code below actually ran T=500/pop=100 --
#            stale-doc/code mismatch. That's why job 103152 (arm a) only got
#            77/500 gens into the FIRST of 5 constraints in 12h before timing
#            out. New numbers are calibrated from that run's observed rate
#            (~90s/gen at pop=100 on this checkpoint) with margin.
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
  TIME_BUDGET=25
  POP_SIZE=50
  echo "=== pareto search | tag=$TAG | FULL budget (T=25, pop=50) ==="
else
  TIME_BUDGET=5
  POP_SIZE=15
  echo "=== pareto search | tag=$TAG | SHORT budget (T=5, pop=15) — smoke test ==="
fi

python -u search_phase.py --ckpt "$CKPT" --mode pareto \
  --constraints 1.0 1.5 2.0 2.5 3.5 \
  --bn_recal_batches 0 \
  --fitness_batches 10 \
  --max_time_budget "$TIME_BUDGET" \
  --population_size "$POP_SIZE" \
  --out "$OUTDIR/search_${TAG}_pareto.csv" || exit 1

echo "=== pareto done: search_${TAG}_pareto.csv ==="
