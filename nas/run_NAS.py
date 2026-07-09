import time
import torch
import random
import argparse

import numpy as np
import pandas as pd

from nas.evolution_finder import EvolutionFinder

from load_model import load_model
from evaluate import evaluate
from load_data import dataset

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_hardware', metavar='TARGET_HARDWARE', help='Target Hardware', required=True)
parser.add_argument('--output', metavar='OUTPUT_PATH', help='The path of output', type=str, required=True)
parser.add_argument('--evo_population', metavar='P', help='The size of population in each generation', type=int, required=True)
parser.add_argument('--evo_generations', metavar='N', help='How many generations of population to be searched', type=int, required=True)
parser.add_argument('--evo_ratio', metavar='R', help='The ratio of networks that are used as parents for next generation', type=float, default=0.25)

args = parser.parse_args()

target_hardware = args.target_hardware
CKPT_PATH = ''

batch_size = 64
# hardware_latency = {'cpu' : [30, 35],
#                     'gpu' : [10, 20, 30, 40]}

hardware_latency = {'cpu' : [3*64, 5*64, 7*64],
                    'gpu' : [0.2*64, 0.35*64, 0.5*65, 0.65*64]}

# Setting random seed
random_seed = 3
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!' % random_seed)

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

if target_hardware == 'gpu':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

model = load_model(CKPT_PATH, dataset)

target_hardware = args.target_hardware
cuda_available = False

if target_hardware == 'cpu':
    device = 'cpu'
    cuda_available = True
else:
    device = 'cuda:0'
    cuda_available = False

use_latency_table = False

latency_constraint = hardware_latency[args.target_hardware][-1]
P = args.evo_population
N = args.evo_generations
r = args.evo_ratio

print(f'Starting Evolutionary Search with P: {P} N: {N} R: {r} and Latency Constraint: {hardware_latency[args.target_hardware]} ms')

params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'arch_mutate_prob': 0.10, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.2, # The ratio of networks that are generated through mutation in generation n >= 2.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
    'verbose': True,
    'dataset': dataset,
    'device': device,
    'supernet': model,
}

finder = EvolutionFinder(**params)
result_lis = []
for latency in hardware_latency[args.target_hardware]:
    print(f"Doing NAS for latency: {latency}")
    best_valids, best_info = finder.run_evolution_search(constraint=latency, verbose=params['verbose'], first=(latency == hardware_latency[args.target_hardware][0]))
    result_lis.append(best_info)
    df = pd.DataFrame(result_lis, columns=['Accuracy', 'Model', 'Latency'])
    df.to_csv(f'{args.output}.csv')
    print(f"Intermediate results saved to {args.output}.csv")

print("NAS Completed!")

df = pd.DataFrame(result_lis, columns=['Accuracy', 'Model', 'Latency'])
df.to_csv(f'{args.output}.csv')
print(f"NAS results saved to f{args.output}_FINAL.csv")
