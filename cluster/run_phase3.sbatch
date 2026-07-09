#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=phase3_%j.out
#
# dynFedNAS — Phase 3: capacity-aware assignment under the SAME bandwidth
# simulation as the Phase 2 characterization sweep (run_sweep.sbatch).
#
# Usage (submit from repo root; args pick deadline and bandwidth estimate):
#   sbatch cluster/run_phase3.sbatch 20           # harsh,    oracle estimate
#   sbatch cluster/run_phase3.sbatch 60           # moderate, oracle estimate
#   sbatch cluster/run_phase3.sbatch 120          # mild,     oracle estimate
#   sbatch cluster/run_phase3.sbatch 20 lagged    # harsh, assign with round r-1's
#                                                 # bandwidth (realistic/stale)
#
# Everything else (model, subnets, schedule, seeds) matches run_sweep.sbatch so
# runs are directly comparable: same bw_seed means the same per-client bandwidth
# trajectories the vanilla TS_all_random runs saw. The comparison that matters:
#   TS_all_random @ T   vs   capacity_aware @ T   (esp. largest-subnet accuracy)
# plus comm/overall_drop_rate (should collapse to ~0, minus best-effort cases)
# and capacity/num_clamped, capacity/max_infeasible.

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate
export WANDB_MODE=offline
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

DEADLINE="$1"
ESTIMATE="${2:-oracle}"
if [ -z "$DEADLINE" ]; then
  echo "ERROR: pass a deadline: 20 | 60 | 120 (and optionally: oracle | lagged)"; exit 1
fi
RUN_NAME="phase3_capacity_${DEADLINE}_${ESTIMATE}"
echo "=== dynFedNAS phase 3 | deadline=${DEADLINE}s | bw_estimate=${ESTIMATE} | run=${RUN_NAME} ==="

python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round 150 \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type capacity_aware \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":120,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha 100 --init_seed 0 \
  --efficient_test --verbose \
  --comm_sim --bw_deadline "$DEADLINE" --bw_seed 0 --bw_estimate "$ESTIMATE" \
  --wandb_project_name dynfednas --wandb_run_name "$RUN_NAME" --wandb_entity none
