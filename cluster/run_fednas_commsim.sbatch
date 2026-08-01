#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=fednas_%j.out
#
# dynFedNAS -- E2c: FedNAS-DARTS under our deadline harness.
#
# ROLE. Not a headline competitor. A large delta against DARTS would mostly
# measure supernet-vs-DARTS, a property we inherited rather than built, and
# cross-family Pareto comparison is the apples-to-oranges problem SuperFedNAS
# itself avoids. This arm exists to show the failure mode is the FAMILY's:
# "dynamic bandwidth breaks FedNAS-DARTS the same way it breaks SuperFedNAS, so
# the problem is the static-communication assumption, not our anchor." That
# claim needs only comm_sim ON vs OFF, which is internal to this arm.
#
# WHY 8 CLIENTS AND NOT 8-OF-20. SuperFedNAS §4.2 runs its FedNAS baseline at
# C=0.4 with K=20 -- our canonical config. This implementation cannot: the
# client index IS the MPI rank (add_local_trained_result(process_id - 1, ...))
# and check_whether_all_receive() waits on every rank, so each worker is a fixed
# client. Matching C=0.4 needs a client-index field in message_define.py, server
# side sampling, update_dataset() on FedNASTrainer (the FedAVGTrainer in this
# same repo has one -- the pattern exists, it just was never applied here), and
# a 20-way partition: ~100 lines across 5 files, and it touches the MPI protocol.
# Deliberately deferred until this cheap version proves the harness works.
#
# Consequence to state in the text, not hide: with 8 clients the bandwidth model
# assigns poor/average/good profiles over 8 rather than 20, so the connectivity
# MIX differs from our other arms. Fine for ON-vs-OFF, which is what we claim;
# NOT fine for putting this arm on the same figure as the SuperFedNAS ladder.
#
# GPU RISK, the reason a smoke comes first: mpirun -np 9 puts 1 server + 8
# clients on ONE GPU. Each process carries its own CUDA context (~0.3-0.5 GB
# before any weights), so ~9 contexts plus 9 DARTS supernets plus optimiser
# state. If this OOMs, options are fewer clients, a smaller batch, or the
# 2080ti/1080ti nodes with more total memory per GPU.
#
# USAGE
#   sbatch cluster/run_fednas_commsim.sbatch <ROUNDS> <on|off> [DEADLINE] [STAGE]
#
#   sbatch cluster/run_fednas_commsim.sbatch 3   on          # smoke, harness on
#   sbatch cluster/run_fednas_commsim.sbatch 3   off         # smoke, control
#   sbatch cluster/run_fednas_commsim.sbatch 150 off         # the ON/OFF pair
#   sbatch cluster/run_fednas_commsim.sbatch 150 on

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=online

ROUNDS="${1:-3}"
COMMSIM="${2:-on}"
DEADLINE="${3:-20}"
STAGE="${4:-search}"

COMM_FLAGS=""
if [ "$COMMSIM" = "on" ]; then
  # Same profile/seed/deadline as the SuperFedNAS arms: tiered_continuous with
  # jitter 0.45 is hardcoded in FedNASAggregator, so only these three vary.
  COMM_FLAGS="--comm_sim --bw_deadline $DEADLINE --bw_seed 0 --bw_jitter_sigma 0.45"
fi

RUN_NAME="fednas_${STAGE}_comm-${COMMSIM}_d${DEADLINE}_r${ROUNDS}"
echo "=== FedNAS E2c | comm_sim=${COMMSIM} deadline=${DEADLINE}s stage=${STAGE} rounds=${ROUNDS} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$SLURM_SUBMIT_DIR/fedml_experiments/distributed/fednas"
hostname > mpi_host_file

# -np 9 = 1 server + 8 clients, matching run_fednas_search.sh. --oversubscribe
# because Slurm allocates 2 CPUs and OpenMPI otherwise refuses to place 9 ranks.
mpirun -np 9 --oversubscribe -hostfile ./mpi_host_file python3 ./main_fednas.py \
  --gpu_server_num 1 --gpu_num_per_server 1 \
  --model darts --dataset cifar10 --partition_method hetero \
  --client_number 8 --comm_round "$ROUNDS" --stage "$STAGE" \
  --epochs 5 --batch_size 64 \
  --frequency_of_the_test 10 \
  $COMM_FLAGS

echo "=== done: $RUN_NAME ==="
OUT="$SLURM_SUBMIT_DIR/fednas_${SLURM_JOB_ID}.out"
grep "\[comm_sim\]" "$OUT" 2>/dev/null | tail -5
