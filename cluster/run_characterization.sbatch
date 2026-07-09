#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=projects
#SBATCH --gpus=1
#SBATCH --output=characterization_%j.out
#
# dynFedNAS — Phase 2: characterize how vanilla SuperFedNAS degrades under
# dynamic bandwidth. Runs the multi-subnet MaxNet config WITH the communication
# simulation (--comm_sim), so slow clients that can't upload their assigned
# subnet in time are dropped. Watch the log for comm/* metrics, especially
# comm/max_subnet_failure_rate (the MaxNet vulnerability) and comm/overall_drop_rate.
#
# BASELINE TO COMPARE AGAINST: the exact same command WITHOUT --comm_sim
# (every update always arrives). Run both, compare accuracy + wall-clock.
#
# CALIBRATING --bw_deadline: it must be neither too loose (no drops) nor too
# tight (everything drops). After a short run, check comm/max_subnet_failure_rate:
# aim for something in ~0.2-0.7. Raise --bw_deadline if too many drops, lower it
# if too few. (Default bandwidth states are 50/10/2 Mbps; see MarkovBandwidth.default.)
#
# Submit from the repo root:  sbatch cluster/run_characterization.sbatch

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/scratch/$USER/dynfednas-venv/bin/activate

export WANDB_MODE=offline
cd "$SLURM_SUBMIT_DIR/fedml_experiments/standalone/superfednas"

python train.py \
  --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 \
  --model ofaresnet50_32x32_10_26 --partition_method hetero \
  --client_num_in_total 20 --client_num_per_round 8 --comm_round 300 \
  --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 \
  --subnet_dist_type TS_all_random \
  --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' \
  --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":240,"init":0.9,"final":0.125}' \
  --frequency_of_the_test 25 --partition_alpha 100 --init_seed 0 \
  --efficient_test --verbose \
  --comm_sim --bw_deadline 20.0 --bw_seed 0 \
  --wandb_project_name dynfednas --wandb_run_name characterization --wandb_entity none
