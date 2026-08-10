#!/bin/bash
#SBATCH --job-name=analysis_cpu
#SBATCH --time=02:00:00
#SBATCH --account=inforet
#SBATCH --partition=interactive-cpu
#SBATCH --gpus=0
#SBATCH --output=analysis_%j.out
#
# dynFedNAS -- CPU-only analysis, on the CPU allocation.
#
# WHY THIS EXISTS. Two problems it fixes at once:
#
# 1. GPU BUDGET. The real constraint on this project is not MaxJobsPU=1, it is
#    GrpTRESMins: the `projects` QOS has a TOTAL budget of 43200 GPU-minutes
#    (720 h), of which 23283 (388 h, 54%) were already spent by 2026-08-09.
#    Analysis work does not need a GPU, so running it on the GPU allocation
#    burns thesis budget for nothing. `inforet` has gpu=0 and its own 6000
#    CPU-minute budget, untouched.
#
# 2. IT RUNS IN PARALLEL. `inforet` is a separate QOS, so its MaxJobsPU=1 is
#    independent of the GPU one -- verified empirically 2026-08-09 by running a
#    GPU job (projects, studgpu-node01) and a CPU job (inforet, studcpu-node01)
#    at the same time. Analysis therefore never costs a slot in the queue that
#    matters. This is the intended use of a CPU allocation, not a workaround.
#
# 3. It stops us running long jobs with nohup on a LOGIN NODE, which is what the
#    deadline sweep did on 2026-08-09. That is fragile (the process dies if the
#    session is recycled), it is antisocial on a shared front end, and the log
#    lands in the /tmp of one specific login node -- ssh round-robins between
#    student-cluster1 and 2, so even finding it again is luck.
#
# LIMITS of this allocation, so a job is not written that cannot fit:
#   cpu=2, mem=16G, gpu=0, walltime capped at 120 m by the allocation (2 h asked here -- the login
#   banner states the limit; an earlier version asked 3.5 h and would have been
#   rejected or truncated),
#   and a 6000 CPU-minute total budget.
# Anything needing torch + a GPU model still fits: building supernet subnets to
# measure payloads is CPU work.
#
# Usage (from the dynfednas_v2 repo root):
#   sbatch cluster/run_analysis_cpu.sbatch analysis/sweep_deadline.py --out analysis/csv/deadline_sweep.csv
#   sbatch cluster/run_analysis_cpu.sbatch analysis/pick_deadline.py --deadlines 20 40 60
#
# The first argument is the script; everything after it is passed through.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: sbatch cluster/run_analysis_cpu.sbatch <script.py> [args...]" >&2
  exit 2
fi

SCRIPT="$1"; shift

echo "=== analysis job: $SCRIPT $* ==="
echo "node: $(hostname)  cpus: ${SLURM_CPUS_ON_NODE:-?}  account: ${SLURM_JOB_ACCOUNT:-?}"
date

source /work/scratch/$USER/dynfednas-venv/bin/activate
cd "$SLURM_SUBMIT_DIR"
python "$SCRIPT" "$@"
rc=$?

date
echo "=== done (exit $rc) ==="

# The reporting block must never decide the job's exit status -- see the note in
# run_fedavg_commsim.sbatch, where `grep | head` took SIGPIPE and marked a
# perfectly good 150-round run as FAILED.
exit $rc
