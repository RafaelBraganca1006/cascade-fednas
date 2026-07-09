#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=sweep_%j.out
#
# dynFedNAS — Phase 2 characterization SWEEP (parameterized).
#
# Usage (submit from repo root; the arg picks the scenario):
#   sbatch cluster/run_sweep.sbatch baseline   # NO comm_sim  -> ideal, nobody drops
#   sbatch cluster/run_sweep.sbatch 20          # comm_sim, deadline 20s  (harsh)
#   sbatch cluster/run_sweep.sbatch 60          # comm_sim, deadline 60s  (moderate)
#   sbatch cluster/run_sweep.sbatch 120         # comm_sim, deadline 120s (mild)
#
# You can submit all four at once; the student cluster runs 1 GPU job at a time,
# so the rest queue (PD) and run in sequence automatically. Each writes sweep_<jobid>.out;
# the mode is echoed at the top of the log and shown as wandb_run_name in the Namespace.
#
# Compare afterwards: accuracy-vs-round per subnet (esp. the largest) and the comm
# metrics (participation, max_subnet_failure_rate) across the four runs.

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=offline
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

MODE="$1"
if [ -z "$MODE" ]; then
  echo "ERROR: pass a mode: baseline | 20 | 60 | 120"; exit 1
fi
if [ "$MODE" = "baseline" ]; then
  COMM_FLAGS=""
  RUN_NAME="sweep_baseline"
else
  COMM_FLAGS="--comm_sim --bw_deadline ${MODE} --bw_seed 0"
  RUN_NAME="sweep_deadline_${MODE}"
fi
echo "=== dynFedNAS sweep | mode=${MODE} | run=${RUN_NAME} ==="

python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round 150 \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type TS_all_random \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha 100 --init_seed 0 \
  --efficient_test --verbose \
  $COMM_FLAGS \
  --wandb_project_name dynfednas --wandb_run_name "$RUN_NAME" --wandb_entity none
