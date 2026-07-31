#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=projects
#SBATCH --qos=projects_4gpus
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --output=par_%j.out
#
# dynFedNAS -- run up to FOUR arms concurrently inside ONE Slurm job.
#
# WHY THIS EXISTS. The account's limit is MaxJobsPU=1: one JOB at a time. It is
# not a limit on GPUs inside that job, and `sacctmgr show assoc` lists a second
# QOS that was never used:
#
#   rbraganca-projects         gres/gpu=1   cpu=32  mem=128G
#   rbraganca-projects_4gpus   gres/gpu=4   cpu=8   mem=112G   <-- this one
#
# Every run so far asked for 1 GPU and 2 CPUs, so the schedule (not the
# hardware) has been the binding constraint the whole time. Four arms in
# parallel is a 4x throughput increase.
#
# WHY IT IS SAFE FOR COMPARABILITY. Nothing about the computation changes --
# same code, same flags, same seeds, one process per GPU via --gpu i. It is
# purely a scheduling change, so arms launched here stay bit-comparable with
# arms run one at a time. (Contrast with raising DataLoader num_workers, which
# would move the augmentation RNG into worker processes and silently change the
# augmentation stream -- that one invalidates cross-run comparability and must
# not be mixed into an existing comparison.)
#
# CPU CAVEAT. This QOS grants 8 CPUs total, i.e. 2 per arm -- the same 2 each
# single-GPU job got, so per-arm speed is unchanged and the gain is entirely in
# aggregate throughput. If a run is ever found to be dataloader-bound, the
# single-GPU `projects` QOS is the one with 32 CPUs, and the two cannot be
# combined.
#
# GPU TYPE. Slurm may mix models (1080ti / 2080ti / 5060ti) across the four
# allocations, which does not affect any accuracy or byte result but does make
# per-arm WALL-CLOCK incomparable. Pin a model with --gres=gpu:<model>:4 when
# the timing itself is the measurement.
#
# USAGE -- one quoted argument per arm, each a full argument list for
# run_selection_probe.sbatch's positional interface:
#
#   sbatch cluster/run_parallel_arms.sbatch \
#     "none 400 0.0 40 0.5 0.45 all_or_nothing off fixed 10 -1" \
#     "none 400 0.0 40 0.5 0.45 nested_blocks on fixed 10 -1"
#
#   # smoke: verify the 4-GPU QOS works at all before a 22h commitment
#   sbatch cluster/run_parallel_arms.sbatch \
#     "none 3 0.0 40 0.5 0.45 nested_blocks on fixed 1 -1" \
#     "none 3 0.0 40 0.5 0.45 nested on fixed 1 -1"
#
# Each arm writes par_<jobid>_arm<N>.out next to this job's own output.

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online

cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

run_arm() {
  local gpu="$1"; shift
  set -- $1
  local SELECTION="${1:-none}"  ROUNDS="${2:-75}"     CORR="${3:-0.0}"
  local DEADLINE="${4:-40}"     ALPHA="${5:-0.5}"     JITTER="${6:-0.45}"
  local TXMODE="${7:-nested}"   MAXNETFIX="${8:-on}"  BLOCKORDER="${9:-fixed}"
  local EVALFREQ="${10:-10}"    PIVOT="${11:--1}"

  local SEL_FLAG=""
  [ "$SELECTION" != "none" ] && SEL_FLAG="--selection_policy $SELECTION"

  local COMM_FLAGS=""
  if [ "$TXMODE" != "none" ]; then
    COMM_FLAGS="--comm_sim --bw_deadline $DEADLINE --bw_seed 0 --tx_mode $TXMODE"
    COMM_FLAGS="$COMM_FLAGS --bw_profile tiered_continuous --bw_jitter_sigma $JITTER"
    COMM_FLAGS="$COMM_FLAGS --bw_data_corr $CORR --bw_estimate lagged"
    [ "$MAXNETFIX" = "on" ] && COMM_FLAGS="$COMM_FLAGS --maxnet_delivery_aware"
  fi

  local NAME="par_${TXMODE}_ord${BLOCKORDER}_piv${PIVOT}_mx${MAXNETFIX}_sel-${SELECTION}_c${CORR}_d${DEADLINE}_a${ALPHA}_j${JITTER}_r${ROUNDS}"
  local OUT="$SLURM_SUBMIT_DIR/par_${SLURM_JOB_ID}_arm${gpu}.out"
  echo "[launch] gpu=$gpu -> $NAME  (log: $OUT)"

  python train.py \
    --gpu "$gpu" --dataset cifar10 --data_dir ./../../../data/cifar10 \
    --model ofaresnet50_32x32_10_26 --partition_method hetero \
    --client_num_in_total 20 --client_num_per_round 8 --comm_round "$ROUNDS" \
    --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
    --subnet_dist_type TS_all_random \
    --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
    --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
    --frequency_of_the_test "$EVALFREQ" --partition_alpha "$ALPHA" --init_seed 0 \
    --efficient_test --verbose --log_per_class --log_inclusion_gap \
    --block_order "$BLOCKORDER" --stream_pivot "$PIVOT" \
    $COMM_FLAGS $SEL_FLAG \
    --wandb_project_name dynfednas --wandb_run_name "$NAME" \
    --wandb_entity rbraganca-eth-z-rich \
    > "$OUT" 2>&1
  echo "[done] gpu=$gpu rc=$? -> $NAME"
}

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  echo "ERROR: pass 1-4 quoted arm argument lists (one per GPU). Got $#." >&2
  exit 2
fi

gpu=0
for arm in "$@"; do
  run_arm "$gpu" "$arm" &
  gpu=$((gpu + 1))
done
wait

echo "=== all arms finished ==="
for f in "$SLURM_SUBMIT_DIR"/par_${SLURM_JOB_ID}_arm*.out; do
  echo "--- $(basename $f)"
  grep "run summary" "$f" 2>/dev/null | tail -1
  grep "shell_cov" "$f" 2>/dev/null | tail -1
done
