python train.py --wandb_project_name cifar10 --wandb_run_name C=0.8-alph=0.6-FedAvg-smallest-subnet --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 100 --client_num_per_round 80 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1}}' --frequency_of_the_test 10 --partition_alpha 0.6 --efficient_test

First Run
'''
python train.py --wandb_project_name cifar10 --wandb_run_name seed0-alph=100-maxnet --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 7 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":1200,"init":0.9,"final":0.125}' --partition_alpha 100 --efficient_test --max_norm 10.0 --init_seed 0
'''
python train.py --wandb_project_name cifar100 --wandb_run_name alph=100-FedDyn-smallest-subnet-wd=5e-3 --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar100 --data_dir ./../../../data/cifar100 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 100 --client_num_per_round 10 --comm_round 501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --feddyn --feddyn_override_wd 0.005
'''
python train.py --wandb_project_name cinic10 --wandb_run_name seed100-alph=0.6-FedAvg-small-subnet --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 6 --dataset cinic10 --data_dir ./../../../data/cinic10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 100 --client_num_per_round 10 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[0,1,0,1],"e":0.14}}' --frequency_of_the_test 10 --partition_alpha 0.6 --efficient_test --init_seed 100 --max_norm 10.0
'''
python train.py --wandb_project_name cifar10-new --wandb_run_name seed0-alph=100-maxnet-train-val-split --wandb_entity awesommesh7-systems-for-ai-laboratory --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":1200,"init":0.9,"final":0.125}' --partition_alpha 100 --efficient_test --max_norm 10.0 --init_seed 0 --use_train_pkl


python train.py --wandb_project_name cifar10 --wandb_run_name seed50-C=0.8-alph=0.6-FedAvg-large-subnet --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 7 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 100 --client_num_per_round 80 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[1,2,1,2],"e":0.22}}' --frequency_of_the_test 10 --partition_alpha 0.6 --efficient_test --init_seed 50 --max_norm 10.0

ps runs:
python train.py --wandb_project_name cifar100-nobn-wd=0 --wandb_run_name alph=100-PS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 4 --dataset cifar100 --data_dir ./../../../data/cifar100 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=100-PS-fedavg_init --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750 --model_ckpt_name best_checkpoint_supernet_2000.pt --run_path flofa/cifar10-cleaned/1956egfp

python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=100-PSw/KD --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750 --kd_ratio 0.5 --teacher_ckpt_name best_checkpoint_supernet_2000.pt --teacher_run_path flofa/cifar10-cleaned/1956egfp
python train.py --wandb_project_name cifar100-nobn-wd=0 --wandb_run_name seed100-alph=100-PSw/KD-fedavg_init --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cifar100 --data_dir ./../../../data/cifar100 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 100 --ps_depth_only 750 --kd_ratio 0.5 --teacher_ckpt_name best_checkpoint_supernet_2000.pt --teacher_run_path flofa/cifar100-nobn-wd=0/23zgzqs1 --model_ckpt_name best_checkpoint_supernet_2000.pt --run_path flofa/cifar100-nobn-wd=0/23zgzqs1

python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=1-PSw/KD-fedavg_init --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 6 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 1 --efficient_test --init_seed 0 --ps_depth_only 750 --kd_ratio 0.5 --teacher_ckpt_name best_checkpoint_supernet_2000.pt --teacher_run_path flofa/cifar10-cleaned/1956egfp --model_ckpt_name best_checkpoint_supernet_2000.pt --run_path flofa/cifar10-cleaned/1956egfp
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name seed100-alph=0.1-PSw/KD-fedavg_init --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 0.1 --efficient_test --init_seed 100 --ps_depth_only 750 --kd_ratio 0.5 --teacher_ckpt_name best_checkpoint_supernet.pt --teacher_run_path flofa/cifar10-cleaned/15c4lbau --model_ckpt_name best_checkpoint_supernet.pt --run_path flofa/cifar10-cleaned/15c4lbau

python train.py --wandb_project_name cinic10 --wandb_run_name alph=100-maxnetcos-maxnorm=10 --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cinic10 --data_dir ./../../../data/cinic10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 100 --client_num_per_round 40 --comm_round 1001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":800,"init":0.9,"final":0.125}' --partition_alpha 100 --efficient_test --init_seed 0 --max_norm 10.0

Client PS runs:
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=100-ClientPSw/KD-fedavg_init --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750 --kd_ratio 0.5 --teacher_ckpt_name best_checkpoint_supernet_2000.pt --teacher_run_path flofa/cifar10-cleaned/1956egfp --model_ckpt_name best_checkpoint_supernet_2000.pt --run_path flofa/cifar10-cleaned/1956egfp --cli_supernet_ps

python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=100-ClientPSw/KD --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 3 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750 --kd_ratio 0.5 --teacher_ckpt_name best_checkpoint_supernet_2000.pt --teacher_run_path flofa/cifar10-cleaned/1956egfp --cli_supernet_ps


dart model runs:
#largest 24 channel 4 layer = 3.66GFLOP
#middle 16 channel 4 layer = 1.82 GFLOP
#smallest 16 channel 2 layer = 0.67 GFLOP

#Largest 8 channel 20 layer = 3.61 GFLOP
#Middle 8 channel 10 layer = 1.74 GFLOP
#Smallest 4 channel 10 layer = 0.64 GFLOP

#Smallest 8 channel 3 layer = 0.47 GFLOP
#Middle 10 channel 8 layer = 1.91 GFLOP
#Largest 12 channel 10 layer = 3.29 GFLOP

#Largest 38 channel 2 layer = 3.18 GFLOP
#Middle 28 channel 2 layer = 1.81 GFLOP
#smallest 16 channel 2 layer = 0.67 GFLOP

#done
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name seed100-alph=100-FedAvg-Darts3_66GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 4 --dataset cifar10 --data_dir ./../../../data/cifar10 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 100 --init_channel_size 24 --darts_layers 4
#done
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name seed100-alph=1-FedAvg-Darts3_66GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 5 --dataset cifar10 --data_dir ./../../../data/cifar10 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 50 --partition_alpha 1 --efficient_test --init_seed 100 --init_channel_size 24 --darts_layers 4
#done
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name seed100-alph=0.1-FedAvg-Darts0_67GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 2501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 50 --partition_alpha 0.1 --efficient_test --init_seed 100 --init_channel_size 16 --darts_layers 2

#seed 0 done
python train.py --wandb_project_name cinic10 --wandb_run_name alph=100-FedAvg-Darts1_81GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 4 --dataset cinic10 --data_dir ./../../../data/cinic10 --model darts --partition_method hetero --client_num_in_total 100 --client_num_per_round 40 --comm_round 1001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 0 --init_channel_size 28 --darts_layers 2
#done
python train.py --wandb_project_name cifar100-nobn-wd=0 --wandb_run_name alph=100-FedAvg-Darts1_81GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 5 --dataset cifar100 --data_dir ./../../../data/cifar100 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 2001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 0 --init_channel_size 28 --darts_layers 2

python train.py --wandb_project_name cinic10 --wandb_run_name alph=100-maxnetcos-maxnorm=10 --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 7 --dataset cinic10 --data_dir ./../../../data/cinic10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 100 --client_num_per_round 40 --comm_round 1001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":800,"init":0.9,"final":0.125}' --partition_alpha 100 --efficient_test --init_seed 0 --max_norm 10.0

python train.py --wandb_project_name cinic10 --wandb_run_name alph=100-FedAvg-Darts3_29GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cinic10 --data_dir ./../../../data/cinic10 --model darts --partition_method hetero --client_num_in_total 100 --client_num_per_round 40 --comm_round 1001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 0 --init_channel_size 12 --darts_layers 10

python train.py --wandb_project_name cifar100-nobn-wd=0 --wandb_run_name seed100-alph=100-FedAvg-Darts8_5GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 0 --dataset cifar100 --data_dir ./../../../data/cifar100 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 2001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 100 --max_norm 10.0 --init_channel_size 64 --darts_layers 2
python train.py --wandb_project_name cifar100-nobn-wd=0 --wandb_run_name seed100-alph=100-FedAvg-Darts191GFLOPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cifar100 --data_dir ./../../../data/cifar100 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 2001 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 100 --max_norm 10.0 --init_channel_size 128 --darts_layers 8
python train.py --wandb_project_name cifar10 --wandb_run_name alph=100-FedAvg-Darts500k --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model darts --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --max_norm 10.0 --init_channel_size 64 --darts_layers 2
python train.py --wandb_project_name cifar10 --wandb_run_name alph=100-FedAvg-Darts500k --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 --model darts --partition_method hetero --client_num_in_total 2 --client_num_per_round 2 --comm_round 1501 --epochs 0 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --max_norm 10.0 --init_channel_size 64 --darts_layers 2
python train.py --wandb_project_name cifar10 --wandb_run_name alph=100-FedAvg-Darts50Mill --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 1 --dataset cifar10 --data_dir ./../../../data/cifar10 --model darts --partition_method hetero --client_num_in_total 2 --client_num_per_round 2 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":[]}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --max_norm 10.0


char tcn runs:
python train.py --wandb_project_name shakespeare --wandb_run_name maxnorm0.05-data-weighted-FedAvg-largest-subnet --wandb_entity flofa --gpu 1 --dataset shakespeare --data_dir ./../../../data/shakespeare --model tcn --partition_method homo --client_num_in_total 150 --client_num_per_round 25 --comm_round 100 --epochs 5 --batch_size 32 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":2,"e":1.0}}' --frequency_of_the_test 5 --efficient_test --max_norm 0.05 --emsize 450 --nhid 450 --levels 3 --weight_dataset --init_seed 0
python train.py --wandb_project_name shakespeare --wandb_run_name seed100-maxnorm0.05-FedAvg-smallest-subnet --wandb_entity flofa --gpu 5 --dataset shakespeare --data_dir ./../../../data/shakespeare --model tcn --partition_method homo --client_num_in_total 40 --client_num_per_round 25 --comm_round 100 --epochs 5 --batch_size 32 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":0,"e":0.2}}' --frequency_of_the_test 5 --efficient_test --max_norm 0.05 --emsize 450 --nhid 450 --skip_train_test --levels 3 --weight_dataset --init_seed 100
python train.py --wandb_project_name shakespeare --wandb_run_name seed100-maxnorm0.05-FedAvg-large-subnet --wandb_entity flofa --gpu 3 --dataset shakespeare --data_dir ./../../../data/shakespeare --model tcn --partition_method homo --client_num_in_total 40 --client_num_per_round 25 --comm_round 100 --epochs 5 --batch_size 32 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":1,"e":1.0}}' --frequency_of_the_test 5 --efficient_test --max_norm 0.05 --emsize 450 --nhid 450 --skip_train_test --levels 3 --weight_dataset --init_seed 100
python train.py --wandb_project_name shakespeare --wandb_run_name seed100-maxnorm0.05-FedAvg-middle-subnet --wandb_entity flofa --gpu 5 --dataset shakespeare --data_dir ./../../../data/shakespeare --model tcn --partition_method homo --client_num_in_total 40 --client_num_per_round 25 --comm_round 100 --epochs 5 --batch_size 32 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":1,"e":0.5}}' --frequency_of_the_test 5 --efficient_test --max_norm 0.05 --emsize 450 --nhid 450 --skip_train_test --levels 3 --weight_dataset --init_seed 100
python train.py --wandb_project_name shakespeare --wandb_run_name seed100-maxnorm0.05-FedAvg-small-subnet --wandb_entity flofa --gpu 6 --dataset shakespeare --data_dir ./../../../data/shakespeare --model tcn --partition_method homo --client_num_in_total 40 --client_num_per_round 25 --comm_round 100 --epochs 5 --batch_size 32 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":1,"e":0.2}}' --frequency_of_the_test 5 --efficient_test --max_norm 0.05 --emsize 450 --nhid 450 --skip_train_test --levels 3 --weight_dataset --init_seed 100

python train.py --wandb_project_name shakespeare --wandb_run_name seed100-min_e0.2-maxnorm0.05-data-weighted-full-dataset-maxnet --wandb_entity flofa --gpu 3 --dataset shakespeare --data_dir ./../../../data/shakespeare --model tcn --partition_method homo --client_num_in_total 150 --client_num_per_round 25 --comm_round 400 --epochs 5 --batch_size 32 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":0,"e":0.2},"1":{"d":1,"e":0.2},"2":{"d":1,"e":0.5},"3":{"d":1,"e":1.0},"4":{"d":2,"e":1.0}}' --frequency_of_the_test 5 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":80,"init":0.9,"final":0.04}' --efficient_test --max_norm 0.05 --emsize 450 --nhid 450 --levels 3 --weight_dataset --ofa_config '{"d":[0,1,2], "e":[0.2, 0.5, 1.0]}' --init_seed 100

#Client PS Runs
python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=100-ClientPS --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 10 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750 --cli_supernet_ps --num_multi_archs 4

python train.py --wandb_project_name cifar10-cleaned --wandb_run_name alph=100-ClientPS-largest_guaruntee --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type PS --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[1,2,1,2],"e":0.22},"4":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --partition_alpha 100 --efficient_test --init_seed 0 --ps_depth_only 750 --cli_supernet_ps --num_multi_archs 4 --PS_with_largest


#TCN runs depth [0, 1, 2]  expand ratio [0.1, 0.2, 0.25, 0.5, 1.0]
#largest: {"0":{"d":2,"e":1.0}} 
#middle: {"0":{"d":1,"e":0.5}} 
#small: {"0":{"d":1,"e":0.2}}
#smallest: {"0":{"d":0,"e":0.1}} 
python train.py --wandb_project_name ptb --wandb_run_name seed100-constLR-20cli-maxnet --wandb_entity flofa --gpu 2 --dataset ptb --data_dir ./../../../data/PTB --model tcn --partition_method homo --client_num_in_total 20 --client_num_per_round 8 --comm_round 201 --epochs 5 --batch_size 16 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":0,"e":0.1},"1":{"d":1,"e":0.2},"2":{"d":1,"e":0.5},"3":{"d":1,"e":1.0},"4":{"d":2,"e":1.0}}' --frequency_of_the_test 5 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":80,"init":0.9,"final":0.125}' --efficient_test --init_seed 100 --max_norm 0.35 --skip_train_test

python train.py --wandb_project_name ptb --wandb_run_name constLR-20cli-FedAvg-large-subnet --wandb_entity flofa --gpu 6 --dataset ptb --data_dir ./../../../data/PTB --model tcn --partition_method homo --client_num_in_total 20 --client_num_per_round 8 --comm_round 101 --epochs 5 --batch_size 16 --client_optimizer sgd --lr 4 --ci 0 --subnet_dist_type static --diverse_subnets '{"0":{"d":1,"e":0.5}}' --frequency_of_the_test 5 --efficient_test --init_seed 0 --max_norm 0.35 --skip_train_test

'''
Completed Runs (FedAvg): 
All Seeds
  cifar10 alpha = [100, 0.6, 0.3]
  cifar100
cinic10 running seed 0 50, 100
500 cli seed 0

'''
Resume function
python resume_flofa_resnet.py --wandb_project_name cifar10-cleaned --wandb_run_name nas_resumed-alph=1-fedavg-maxnetcos --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --run_path flofa/cifar10-cleaned/8ka6r39j --model_ckpt_name finished_checkpoint_data_1000.pt --resume_round 1001 --comm_round 1501 --frequency_of_the_test 50 --verbose --diverse_subnets '{"0":{"d":[2,2,2,2],"e":0.25}}'
'''
python resume_flofa_resnet.py --wandb_project_name cifar10-cleaned --wandb_run_name test_nas_resumed1-alph=1-fedavg-maxnetcos --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 6 --run_path flofa/cifar10-cleaned/8ka6r39j --model_ckpt_name finished_checkpoint_data_1000.pt --resume_round 1001 --comm_round 1501 --frequency_of_the_test 2 --verbose --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' --ckpt_subnets '[{"d":[2,2,2,2],"e":0.25}]' --custom_config '{
  "wandb_version": 1,
  "_wandb": {
    "desc": null,
    "value": {
      "cli_version": "0.10.30",
      "framework": "torch",
      "is_jupyter_run": false,
      "is_kaggle_kernel": false,
      "python_version": "3.8.5",
      "t": {
        "1": [
          1
        ],
        "2": [
          1
        ],
        "3": [
          1,
          3
        ],
        "4": "3.8.5",
        "5": "0.10.30",
        "8": [
          5
        ]
      }
    }
  },
  "batch_size": {
    "desc": null,
    "value": 64
  },
  "best_model_freq": {
    "desc": null,
    "value": 1000
  },
  "bn_gamma_zero_init": {
    "desc": null,
    "value": false
  },
  "ci": {
    "desc": null,
    "value": 0
  },
  "cli_subnet_track": {
    "desc": null,
    "value": null
  },
  "cli_supernet": {
    "desc": null,
    "value": false
  },
  "client_num_in_total": {
    "desc": null,
    "value": 20
  },
  "client_num_per_round": {
    "desc": null,
    "value": 8
  },
  "client_optimizer": {
    "desc": null,
    "value": "sgd"
  },
  "comm_round": {
    "desc": null,
    "value": 1801
  },
  "data_dir": {
    "desc": null,
    "value": "./../../../data/cifar10"
  },
  "dataset": {
    "desc": null,
    "value": "cifar10"
  },
  "diverse_subnets": {
    "desc": null,
    "value": {
      "0": {
        "d": [
          0,
          0,
          0,
          0
        ],
        "e": 0.1
      },
      "1": {
        "d": [
          0,
          1,
          0,
          1
        ],
        "e": 0.14
      },
      "2": {
        "d": [
          1,
          1,
          1,
          1
        ],
        "e": 0.18
      },
      "3": {
        "d": [
          2,
          2,
          2,
          2
        ],
        "e": 0.25
      }
    }
  },
  "efficient_test": {
    "desc": null,
    "value": true
  },
  "epochs": {
    "desc": null,
    "value": 5
  },
  "frequency_of_the_test": {
    "desc": null,
    "value": 50
  },
  "gpu": {
    "desc": null,
    "value": 5
  },
  "init_seed": {
    "desc": null,
    "value": 100
  },
  "inplace_kd": {
    "desc": null,
    "value": false
  },
  "kd_ratio": {
    "desc": null,
    "value": 0
  },
  "kd_type": {
    "desc": null,
    "value": "ce"
  },
  "largest_step_more": {
    "desc": null,
    "value": false
  },
  "largest_subnet_wd": {
    "desc": null,
    "value": 0
  },
  "lr": {
    "desc": null,
    "value": 0.1
  },
  "lr_schedule": {
    "desc": null,
    "value": null
  },
  "model": {
    "desc": null,
    "value": "ofaresnet50_32x32_10_26"
  },
  "model_checkpoint_freq": {
    "desc": null,
    "value": 500
  },
  "model_ckpt_name": {
    "desc": null,
    "value": null
  },
  "multi": {
    "desc": null,
    "value": false
  },
  "multi_disable_rest_bn": {
    "desc": null,
    "value": false
  },
  "multi_drop_largest": {
    "desc": null,
    "value": false
  },
  "num_multi_archs": {
    "desc": null,
    "value": 1
  },
  "ofa_config": {
    "desc": null,
    "value": null
  },
  "optim_step_more": {
    "desc": null,
    "value": false
  },
  "partition_alpha": {
    "desc": null,
    "value": 1
  },
  "partition_method": {
    "desc": null,
    "value": "hetero"
  },
  "reset_bn_sample_size": {
    "desc": null,
    "value": 0.2
  },
  "reset_bn_stats": {
    "desc": null,
    "value": false
  },
  "reset_bn_stats_test": {
    "desc": null,
    "value": false
  },
  "resume_round": {
    "desc": null,
    "value": -1
  },
  "run_path": {
    "desc": null,
    "value": null
  },
  "skip_train_largest": {
    "desc": null,
    "value": false
  },
  "subnet_dist_type": {
    "desc": null,
    "value": "TS_all_random"
  },
  "teacher_ckpt_name": {
    "desc": null,
    "value": null
  },
  "teacher_run_path": {
    "desc": null,
    "value": null
  },
  "use_bn": {
    "desc": null,
    "value": false
  },
  "verbose": {
    "desc": null,
    "value": true
  },
  "verbose_test": {
    "desc": null,
    "value": false
  },
  "wandb_entity": {
    "desc": null,
    "value": "flofa"
  },
  "wandb_project_name": {
    "desc": null,
    "value": "cifar10-cleaned"
  },
  "wandb_run_name": {
    "desc": null,
    "value": "seed100-alph=1-maxnetcos"
  },
  "wandb_watch": {
    "desc": null,
    "value": true
  },
  "wandb_watch_freq": {
    "desc": null,
    "value": 250
  },
  "warmup_init_lr": {
    "desc": null,
    "value": 0
  },
  "warmup_rounds": {
    "desc": null,
    "value": 0
  },
  "wd": {
    "desc": null,
    "value": 0
  },
  "weighted_avg_schedule": {
    "desc": null,
    "value": {
      "final": 0.125,
      "init": 0.9,
      "num_steps": 1200,
      "type": "maxnet_cos_all_subnet"
    }
  }
}'
'''

# Test functionality
'''
python test_flofa_resnet.py --wandb_project_name test_nas --wandb_run_name verify3-alph=100-maxnetcos --wandb_entity flofa --gpu 4 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --batch_size 64 --partition_alpha 100 --init_seed 100 --run_path flofa/cifar10-cleaned/1pqzuk1l --model_ckpt_name best_checkpoint_supernet_2000.pt --test_subnets '[{"d":[0,0,0,1],"e":0.1}, {"d":[2,2,2,2],"e":0.25}]'
'''

# NAS FUNCTIONALITY
'''
python test_flofa_resnet.py --wandb_project_name test_nas --wandb_run_name 2nd-NAS4-seed0-alph=1-maxnetcos --wandb_entity flofa --gpu 5 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --batch_size 64 --partition_alpha 1 --init_seed 0 --run_path flofa/cifar10-cleaned/1ujgbi87 --model_ckpt_name best_checkpoint_supernet.pt --nas --nas_constraints [4] --max_time_budget 100 --population 250 --parent_ratio 0.25 --mutation_ratio 0.6
'''
python test_flofa_resnet.py --wandb_project_name test_nas --wandb_run_name test2-proofOfConceptNAS-seed0-alph=100-maxnetcos --wandb_entity flofa --gpu 2 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --batch_size 64 --partition_alpha 100 --init_seed 0 --run_path flofa/cifar10-cleaned/3vn2tjv1 --model_ckpt_name best_checkpoint_supernet.pt --nas --nas_constraints 1,2,3,4 --max_time_budget 20 --population 4 --test_subnets '[{"d":[0,0,0,0],"e":0.1}, {"d":[0,1,0,1],"e":0.14}, {"d":[0,1,1,1],"e":0.14}, {"d":[1,1,1,1],"e":0.18}, {"d":[1,2,1,2],"e":0.22}, {"d":[1,2,2,2],"e":0.22}]'
'''
from ofa.utils import flops_counter as fp
from fedml_api.standalone.flofa.elastic_nn.ofa_resnets_32x32_10_26 import (
     OFAResNets32x32_10_26,
     ResNets32x32_10_26,
 )
supernet = OFAResNets32x32_10_26()
supernet.set_active_subnet(**{"d":[1,1,1,1], "e": 0.18})
smallest_subnet = supernet.get_active_subnet()
fp.profile(smallest_subnet, (1,3,32, 32))
'''

# Multi Seed Test and Save Functionality
'''
python test_flofa_resnet.py --wandb_project_name cinic10 --wandb_run_name multiseedtest-cinic10-maxnetcos --wandb_entity flofa --gpu 7 --dataset cinic10 --data_dir ./../../../data/cinic10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --batch_size 64 --partition_alpha 100 --multi_seed_test --seed_list '[0, 50, 100]' --multi_seed_run_paths '["flofa/cinic10/34581hez", "flofa/cinic10/o1wxhjik", "flofa/cinic10/f3jiocso"]' --multi_seed_model_ckpt_names '["best_checkpoint_supernet_1000.pt", "best_checkpoint_supernet_1000.pt", "best_checkpoint_supernet_1000.pt"]' --multi_seed_test_subnets '[{"d":[0,0,0,1],"e":0.14}, {"d":[0,1,1,1],"e":0.14}, {"d":[1,1,1,2],"e":0.22}, {"d":[1,2,1,2],"e":0.22}, {"d":[1,2,2,2],"e":0.22}]' --save_file ../../../pareto_plotting/MultiSeedCSVs/alph100_cinic10_maxnetcos.csv
'''
'''
python test_flofa_resnet.py --wandb_project_name test_nas --wandb_run_name multiseedtest-alph=100-maxnetcos --wandb_entity flofa --gpu 3 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --batch_size 64 --partition_alpha 100 --multi_seed_test --seed_list '[0, 50, 100]' --multi_seed_run_paths '["flofa/cifar10-cleaned/3vn2tjv1", "flofa/cifar10-cleaned/1y1tkuw3", "flofa/cifar10-cleaned/1pqzuk1l"]' --multi_seed_model_ckpt_names '["best_checkpoint_supernet.pt", "best_checkpoint_supernet.pt", "best_checkpoint_supernet_2000.pt"]' --multi_seed_test_subnets '[{"d":[0,0,0,1],"e":0.14}, {"d":[0,1,1,1],"e":0.14}, {"d":[1,1,1,2],"e":0.22}, {"d":[1,2,1,2],"e":0.22}, {"d":[1,2,2,2],"e":0.22}, {"d":[2,2,2,2],"e":0.25}]' --save_file ../../../pareto_plotting/MultiSeedCSVs/alph100_cifar10_maxnetcos.csv
'''
# Dry Run
'''
python main_flofa_resnet.py --wandb_project_name cifar10-cleaned --wandb_run_name DRYRUN-seed100-alph=100-maxnetcos --wandb_entity flofa --wandb_watch --wandb_watch_freq 250 --gpu 2 --dataset cifar10 --data_dir ./../../../data/cifar10 --model ofaresnet50_32x32_10_26 --partition_method hetero --client_num_in_total 20 --client_num_per_round 8 --comm_round 1501 --epochs 5 --batch_size 64 --client_optimizer sgd --lr 0.1 --ci 0 --subnet_dist_type TS_all_random --diverse_subnets '{"0":{"d":[0,0,0,0],"e":0.1},"1":{"d":[0,1,0,1],"e":0.14},"2":{"d":[1,1,1,1],"e":0.18},"3":{"d":[2,2,2,2],"e":0.25}}' --frequency_of_the_test 50 --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":1200,"init":0.9,"final":0.125}' --partition_alpha 100 --efficient_test --verbose --init_seed 100 --dry_run
'''
