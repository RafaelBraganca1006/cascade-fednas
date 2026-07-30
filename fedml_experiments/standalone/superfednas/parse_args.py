import json


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="Pass on multiple networks to a single client",
    )

    parser.add_argument(
        "--skip_train_largest",
        action="store_true",
        help="Pass on multiple networks to a single client",
    )

    parser.add_argument(
        "--multi_drop_largest",
        action="store_true",
        help="(Needs to be reworked!) drops largest model and smallest to allow single middle model to be aggregated only",
    )

    parser.add_argument(
        "--cli_supernet",
        action="store_true",
        help="Pass on entire supernetwork to a single client",
    )

    parser.add_argument(
        "--cli_supernet_ps",
        action="store_true",
        help="Pass on entire supernetwork to a single client and perform PS on each client",
    )

    parser.add_argument(
        "--inplace_kd", action="store_true", help="In place knowledge distillation",
    )

    parser.add_argument(
        "--multi_disable_rest_bn",
        action="store_true",
        help="Multi setting Disable BN for rest of the networks apart from the largest one",
    )

    parser.add_argument(
        "--num_multi_archs",
        type=int,
        default=1,
        metavar="K",
        help="Number of middle architectures for multi net run",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Project Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Run Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="superfednas",
        metavar="N",
        help="helps in creating a project under teams or personal username",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./../../../data/cifar10",
        help="data directory",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="homo",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=1,
        metavar="PA",
        help="partition alpha (default: 1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--warmup_init_lr",
        type=float,
        default=0,
        help="warmup learning rate init (default: 0)",
    )

    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=0,
        help="Number of rounds to perform LR warmup",
    )

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0)

    parser.add_argument(
        "--largest_subnet_wd",
        help="weight decay parameter for largest subnet",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=20,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=8,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=1000,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=50,
        help="the frequency of the algorithms",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument(
        "--diverse_subnets",
        type=json.loads,
        default='{"0":{"d":[2, 2, 2, 2, 2],"w0_avg":1,"wf_avg":1}}',
        help="Subnets participating in training and possibly their starting and final weights during averaging",
    )

    parser.add_argument(
        "--ckpt_subnets",
        type=json.loads,
        default=None,
        help="List of subnets used to checkpoint best model on interval",
    )

    parser.add_argument(
        "--use_bn", action="store_true", help="Use batchnorm",
    )

    parser.add_argument(
        "--bn_gamma_zero_init",
        action="store_true",
        default=False,
        help="Sets the learnable scaling coefficient γ = 0 in the last Batch Normalization layer of each residual block",
    )

    parser.add_argument(
        "--reset_bn_stats",
        action="store_true",
        help="Resets bn mean and variance before test and training",
        default=False,
    )

    parser.add_argument(
        "--reset_bn_stats_test",
        action="store_true",
        help="Resets bn mean and variance before testing",
    )

    parser.add_argument(
        "--efficient_test", action="store_true", help="Test efficiently",
    )

    parser.add_argument(
        "--reset_bn_sample_size",
        type=float,
        default=0.2,
        help="percentage of local train data to use to reset bn stats",
    )

    parser.add_argument(
        "--subnet_dist_type",
        type=str,
        default="all_random",
        choices=[
            "static",
            "dynamic",
            "all_random",
            "compound",
            "sandwich_all_random",
            "sandwich_compound",
            "TS_all_random",
            "TS_compound",
            "max_sample_count",
            "multi_sandwich",
            "TS_KD",
            "PS",
            "policy_assign",
            "capacity_aware",
        ],
        help="Subnetwork selection strategy for every round",
    )

    parser.add_argument(
        "--weighted_avg_schedule",
        type=json.loads,
        default=None,
        help="Dynamic Weighted Average Hyper-parameters",
    )

    parser.add_argument(
        "--model_ckpt_name", type=str, default=None, help="Name of model to load",
    )

    parser.add_argument(
        "--run_path", type=str, default=None, help="Run path of model to load",
    )

    parser.add_argument(
        "--cli_subnet_track",
        type=json.loads,
        default=None,
        help="Subnet tracker state for resuming run",
    )

    parser.add_argument(
        "--teacher_ckpt_name",
        type=str,
        default=None,
        help="Name of teacher model to load",
    )

    parser.add_argument(
        "--teacher_run_path",
        type=str,
        default=None,
        help="Run path of teacher model to load",
    )

    parser.add_argument(
        "--ofa_config", type=json.loads, default=None, help="OFA ResNet configuration",
    )

    parser.add_argument(
        "--kd_ratio",
        type=float,
        default=0,
        help="knowledge Distillation using pretrained model",
    )

    parser.add_argument(
        "--kd_type",
        type=str,
        default="ce",
        choices=["ce", "mse"],
        help="knowledge Distillation using pretrained model",
    )

    parser.add_argument(
        "--lr_schedule", type=json.loads, default=None, help="Weighted random"
    )

    parser.add_argument(
        "--wandb_watch", action="store_true", help="Watch parameters on wandb",
    )

    parser.add_argument(
        "--wandb_watch_freq", type=int, default=100, help="wandb watch frequency",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just simulate distribution of subnetworks and log number of parameters and gflops of each subnet",
    )

    parser.add_argument(
        "--model_checkpoint_freq",
        type=int,
        default=500,
        help="frequency at which model is checkpointed",
    )

    parser.add_argument(
        "--best_model_freq",
        type=int,
        default=1000,
        help="intervals at which to track best model",
    )

    parser.add_argument(
        "--resume_round", type=int, default=-1, help="round to resume training at",
    )

    parser.add_argument(
        "--custom_config",
        type=json.loads,
        default=None,
        help="Allows user to specify custom config.yaml for crashed runs or runs missing config.yaml",
    )

    parser.add_argument(
        "--optim_step_more",
        action="store_true",
        help="Take optimizer step after each gradient in cli supernet",
    )

    parser.add_argument(
        "--largest_step_more",
        action="store_true",
        help="Take optimizer step after largest subnet gradient in cli supernet",
    )

    parser.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help="Initial seed for intializing model weights and partitioning dataset amongst clients",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Log in more detail",
    )

    parser.add_argument(
        "--verbose_test", action="store_true", help="Log testing in more detail",
    )

    parser.add_argument(
        "--feddyn", action="store_true", help="Use FedDyn Optimization",
    )

    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.01,
        help="FedDyn optimization hyperparameter alpha. Generally one of [0.1, 0.01, 0.001]",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="optimization gradient clipping max_norm",
    )
    parser.add_argument(
        "--feddyn_max_norm",
        type=float,
        default=10,
        help="FedDyn optimization gradient clipping max_norm",
    )
    parser.add_argument(
        "--feddyn_no_wd_modifier",
        action="store_true",
        help="Don't use feddyn modified wd",
    )

    parser.add_argument(
        "--feddyn_override_wd",
        type=float,
        default=-1,
        help="Override feddyn wd if > 0",
    )

    parser.add_argument(
        "--mod_wd_dyn",
        action="store_true",
        help="Use FedDyn modified wd in non-feddyn runs",
    )

    parser.add_argument(
        "--weight_dataset",
        action="store_true",
        help="Include dataset size into weighted averaging",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.45,
        help="Elastic TCN dropout",
    )

    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.25,
        help="Elastic TCN embedding dropout",
    )

    parser.add_argument(
        "--ksize",
        type=int,
        default=3,
        help="Elastic TCN kernel size",
    )

    parser.add_argument(
        "--emsize",
        type=int,
        default=600,
        help="Elastic TCN size of word embeddings (default: 600)",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Elastic TCN # of levels (default: 4)",
    )

    parser.add_argument(
        "--nhid",
        type=int,
        default=600,
        help="Elastic TCN number of hidden units per layer (default: 600)",
    )

    parser.add_argument(
        "--tied",
        action="store_false",
        help="Include dataset tie the encoder-decoder weights (default: True)",
    )

    parser.add_argument(
        "--validseqlen",
        type=int,
        default=40,
        help="Elastic TCN valid sequence length (default: 40)",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=80,
        help="Elastic TCN total sequence length, including effective history (default: 80)",
    )

    parser.add_argument(
        "--skip_train_test",
        action="store_true",
        help="skip train ppl calculations",
    )

    parser.add_argument(
        "--top_k_maxnet",
        type=int,
        default=1,
        help="How many maximum subnets to sample per round. Must have top_k_maxnet+bottom_k_maxnet <= num clients per round",
    )

    parser.add_argument(
        "--bottom_k_maxnet",
        type=int,
        default=1,
        help="How many minimum subnets to sample per round. Must have top_k_maxnet+bottom_k_maxnet <= num clients per round",
    )
    parser.add_argument(
        "--init_channel_size",
        type=int,
        default=128,
        help="Initial number of channels in convolution of darts model",
    )
    parser.add_argument(
        "--darts_layers",
        type=int,
        default=20,
        help="Number of layers in darts model",
    )
    parser.add_argument(
        "--ps_depth_only",
        type=int,
        default=750,
        help="Number of rounds to only sample depth elastic random models",
    )

    parser.add_argument(
        "--PS_with_largest",
        action="store_true",
        help="During client supernet training using PS, select the largest subnetwork once and sample k-1 random subnets based on phase",
    )

    parser.add_argument(
        "--use_train_pkl", action="store_true", help="Use train dataset generated after splitting original train dataset into train and val datasets",
    )

    # ---- dynFedNAS: communication simulation (Phase 1 harness) ----
    parser.add_argument(
        "--comm_sim",
        action="store_true",
        help="Enable dynamic-bandwidth communication simulation. When on, a client's "
        "update is dropped from aggregation if it cannot be uploaded within --bw_deadline "
        "given its current (time-varying) bandwidth.",
    )
    parser.add_argument(
        "--bw_deadline",
        type=float,
        default=20.0,
        help="Per-round upload deadline in seconds (the round timeout budget).",
    )
    parser.add_argument(
        "--bw_model",
        type=str,
        default="markov",
        choices=["markov"],
        help="Bandwidth source for the communication simulation.",
    )
    parser.add_argument(
        "--bw_seed",
        type=int,
        default=0,
        help="Seed for the (per-client) bandwidth model, for reproducibility.",
    )
    parser.add_argument(
        "--bw_estimate",
        type=str,
        default="oracle",
        choices=["oracle", "lagged"],
        help="Bandwidth estimate the capacity_aware sampler assigns with: this "
        "round's true bandwidth (oracle, upper bound) or the previous round's "
        "(lagged, realistic — the estimate can be stale at upload time).",
    )
    parser.add_argument(
        "--tx_mode",
        type=str,
        default="all_or_nothing",
        choices=["all_or_nothing", "nested", "nested_shard", "nested_blocks"],
        help="Transmission model. all_or_nothing: the whole update fits within "
        "the deadline or it is dropped (legacy). nested: the upload is a "
        "shell-ordered stream (S, M\\S, L\\M, ...) the deadline truncates — "
        "every complete shell prefix is a valid smaller-tier update (anytime). "
        "nested_shard: nested + top-shell sharding across clients. "
        "nested_blocks (DIRECTION §3.1c): the definitive design — same shell "
        "ORDER, but the physical unit is (shell, block), so the deadline cuts "
        "at block granularity instead of losing a whole shell. A client whose "
        "budget is below the smallest tier still contributes whole blocks "
        "(collective rule) instead of delivering nothing: measured on the E6b "
        "bandwidth distribution, dead slots 34.8%% -> ~0%% and waste 23.6%% -> "
        "~2%%. `nested` stays frozen as the reference for the archived arms.",
    )
    parser.add_argument(
        "--no_shard_fallback",
        action="store_true",
        help="nested_shard only: disable the sub-floor fallback (clients whose "
        "budget fits no complete tier upload whole blocks of their own subnet "
        "instead of dropping). On by default: zero-drop by construction.",
    )
    parser.add_argument(
        "--shard_carriers",
        type=int,
        default=2,
        help="nested_shard only: number of clients per round designated as "
        "shard carriers (train the full supernet, upload assigned supernet "
        "blocks in coverage-debt order). Carriers are picked by lagged "
        "bandwidth (round r-1, no oracle).",
    )
    parser.add_argument(
        "--assignment_policy",
        type=str,
        default="none",
        choices=["none", "greedy", "lyapunov", "bandit"],
        help="Only used when --subnet_dist_type policy_assign (DIRECTION §3.2, "
        "retargeted 2026-07-22), and requires --tx_mode nested. Which "
        "AssignmentPolicy decides each client's tier under nested "
        "transmission. 'greedy' is the lambda=0 optimum (Lemma 1, no E3 "
        "dependency); 'lyapunov'/'bandit' are both fully implemented and "
        "need --policy_lambda from the E3 ablation (see comm_sim/policies/).",
    )
    parser.add_argument(
        "--selection_policy",
        type=str,
        default="none",
        choices=["none", "oracle_bandwidth", "coverage"],
        help="E6b Stage 1 leverage probes (EXPERIMENTS.md, DIRECTION §3.2b): "
        "WHICH clients participate each round, the decision _client_sampling "
        "makes uniformly at random today. 'none' = the legacy uniform draw, "
        "bit-identical to every archived run. 'oracle_bandwidth' pushes the "
        "bandwidth signal to its extreme (top-K by true current bandwidth); "
        "'coverage' pushes the staleness signal to its extreme (maximise the "
        "minimum delivered-coverage). These are NOT ceilings/upper bounds — "
        "oracle_bandwidth can easily land BELOW blind under non-IID, and that "
        "is an informative result. They measure whether the 'who participates' "
        "axis carries any signal at all before an RL is built on it.",
    )
    parser.add_argument(
        "--policy_lambda",
        type=float,
        default=None,
        help="Cross-context penalty rate for the lyapunov/bandit assignment "
        "policies (DIRECTION §3.2) -- set from the E3 gradient-context "
        "ablation result. Required (no default) for lyapunov/bandit; unused "
        "by greedy.",
    )
    parser.add_argument(
        "--maxnet_delivery_aware",
        action="store_true",
        help="DIRECTION §3.3 fix (2026-07-23): scale a client's MaxNet "
        "aggregation weight by how much of its intended (trained) payload "
        "actually arrived, instead of applying the full pre-round slot "
        "weight (which can be up to beta=0.9 early in training) to a small "
        "truncated slice. nested/nested_shard's shell-truncation path only "
        "(same scope as --nested_context_ablation). Off by default -- "
        "preserves comparability with all prior results.",
    )
    parser.add_argument(
        "--nested_context_ablation",
        action="store_true",
        help="E3 gradient-context ablation (DIRECTION §3.1a). nested/nested_shard "
        "only: discard cross-context deliveries (client trained a larger config "
        "alpha, deadline truncated to a smaller beta) instead of aggregating the "
        "sliced beta-update. Compare per-tier accuracy against a matched run with "
        "this flag off (same bandwidth seed) to test whether small tiers are "
        "poisoned by larger-context gradients. Off by default (normal nested "
        "behavior unaffected).",
    )
    parser.add_argument(
        "--bw_profile",
        type=str,
        default="uniform",
        choices=["uniform", "tiered", "tiered_continuous"],
        help="Connectivity heterogeneity (E4). uniform: every client has the "
        "same Markov dynamics (legacy — no client is persistently "
        "disadvantaged). tiered: persistent per-client profiles "
        "(poor/average/good dwell dynamics over the same states). "
        "tiered_continuous (E6b, 2026-07-26): tiered + log-normal spread "
        "WITHIN each state, so the budget is continuous and the whole tier "
        "ladder is reachable. With three discrete states there are only three "
        "possible budgets at any deadline, so at least one tier is always "
        "unreachable — the structural cause of the degenerate assignment "
        "problem found in the policy audit (DIRECTION §3.2b). Mean bandwidth "
        "is unchanged vs tiered; only variance differs.",
    )
    parser.add_argument(
        "--bw_jitter_sigma",
        type=float,
        default=0.45,
        help="Log-normal within-state spread for --bw_profile "
        "tiered_continuous. Chosen by analysis/pick_deadline.py: at 0.45 with "
        "--bw_deadline 40 every tier is reachable and ~35%% of participation "
        "slots stay structurally dead (the lever the selection policy "
        "reclaims). Below ~0.35 the M tier dies again; above ~0.6 the good "
        "state's tail stops being physical. Exploratory, to be calibrated "
        "against the Irish traces' within-regime variance when E9 lands.",
    )
    parser.add_argument(
        "--bw_data_corr",
        type=float,
        default=0.0,
        help="tiered profile only: correlation strength in [0,1] between a "
        "client's connectivity tier and its label distribution. 0 = random "
        "assignment; 1 = clients concentrating low-index classes get the "
        "poor tier (digital-divide stress test, DIRECTION §5).",
    )
    parser.add_argument(
        "--log_per_class",
        action="store_true",
        help="Log per-class test accuracy of each tier probe at every eval "
        "checkpoint (class x method heatmap data, E4).",
    )
    parser.add_argument(
        "--block_order",
        type=str,
        default="fixed",
        choices=["fixed", "random"],
        help="Within-shell send order for --tx_mode nested_blocks. 'fixed' "
        "(default) sends blocks by index, which the mechanism turns into a "
        "SYSTEMATIC bias: the deadline always cuts the same high-index blocks "
        "of every shell, so they are chronically under-trained. 'random' "
        "reshuffles per (round, client) -- free, since blocks inside a shell "
        "are independent -- and measures what that bias is worth before a "
        "learned ordering policy is built on top (DIRECTION 3.2c).",
    )
    parser.add_argument(
        "--stream_pivot",
        type=int,
        default=-1,
        help="Cohort-complementary streaming for --tx_mode nested_blocks "
        "(DIRECTION 3.1d). -1 (default) = OFF, today's pure shell-major stream. "
        "p >= 0 keeps shells 0..p-1 shell-major (so a client that affords them "
        "still delivers a valid subnet update, stem/classifier included) and "
        "switches shells p.. to BLOCK-major: one block complete at full width, "
        "then the next. Combined with a per-client rotation offset (round + "
        "client index, applied automatically), different clients cover "
        "different blocks and the COHORT covers the whole supernet while every "
        "client still sends the same number of bytes -- nobody is diverted to "
        "carrier duty, which is what made shell sharding lose. Motivation: "
        "shell-major spends a round's bytes on 8 copies of the same low region "
        "while the XL-exclusive top shell, sent last, gets ~27% coverage "
        "(analysis/d3_coverage.py), i.e. most bytes go where the marginal "
        "value is lowest. p=1 is the natural setting: guarantee the shared "
        "core, complement everything above it.",
    )
    parser.add_argument(
        "--log_inclusion_gap",
        action="store_true",
        help="Log the accuracy gap between the worst- and best-connectivity "
        "client groups, on their OWN local test data (DIRECTION §2 secondary "
        "claim). Needs a per-client eval pass that --efficient_test skips, so "
        "it is opt-in. REQUIRED on every E6b selection run: a policy that wins "
        "mean accuracy while widening this gap is publishing the digital-divide "
        "pathology as its solution. No-op unless --bw_profile is tiered*.",
    )

    return parser

def add_args_test(parser):
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Project Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="superfednas",
        metavar="N",
        help="Run Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="superfednas",
        metavar="N",
        help="helps in creating a project under teams or personal username",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./../../../data/cifar10",
        help="data directory",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="homo",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=1,
        metavar="PA",
        help="partition alpha (default: 1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=20,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument(
        "--model_ckpt_name", type=str, default=None, help="Name of model to load",
    )

    parser.add_argument(
        "--run_path", type=str, default=None, help="Run path of model to load",
    )

    parser.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help="Initial seed for intializing model weights and partitioning dataset amongst clients",
    )

    parser.add_argument(
        "--test_subnets",
        type=json.loads,
        default=None,
        help="Subnets to test as a list",
    )

    parser.add_argument(
        "--nas", action="store_true", help="Perform NAS",
    )

    parser.add_argument(
        "--nas_constraints",
        type=json.loads,
        default=None,
        help="List of constraints to be used during nas",
    )

    parser.add_argument(
        "--mutate_prob", type=float, default=0.1, help="nas mutation probability",
    )

    parser.add_argument(
        "--population_size", type=int, default=100, help="nas population size",
    )

    parser.add_argument(
        "--max_time_budget", type=int, default=500, help="nas max time taken to sample",
    )

    parser.add_argument(
        "--parent_ratio", type=float, default=0.25, help="nas parent ratio",
    )

    parser.add_argument(
        "--mutation_ratio", type=float, default=0.5, help="nas mutation ratio",
    )

    parser.add_argument(
        "--multi_seed_test",
        action="store_true",
        help="Perform test on multiple seed and save results for plotting",
    )

    parser.add_argument(
        "--seed_list",
        type=json.loads,
        default=None,
        help="List of init seed to be used during multi seed test",
    )

    parser.add_argument(
        "--multi_seed_run_paths",
        type=json.loads,
        default=None,
        help="run path for model to test as a list corresponding to seed_list",
    )

    parser.add_argument(
        "--multi_seed_model_ckpt_names",
        type=json.loads,
        default=None,
        help="names of model to load from run paths",
    )

    parser.add_argument(
        "--multi_seed_test_subnets",
        type=json.loads,
        default=None,
        help="Subnets to test as a list for multi seed testing",
    )

    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="path and name of csv to save test data",
    )

    parser.add_argument(
        "--feddyn", action="store_true", help="Use FedDyn Optimization",
    )

    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.1,
        help="FedDyn optimization hyperparameter alpha. Generally one of [0.1, 0.01, 0.001]",
    )

    return parser
