# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
import numpy as np
from tqdm import tqdm
from evaluate import evaluate

__all__ = ["EvolutionFinder"]


class EvolutionFinder:
    def __init__(self, **kwargs):
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.init_arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 200)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

        self.dataset = kwargs.get("dataset", None)
        assert self.dataset is not None

        self.device = kwargs.get("device", None)
        assert self.device is not None

        self.supernet = kwargs.get("supernet", None)
        self.model = self.supernet.model
        assert self.supernet is not None


    def accuracy_efficiency_predictor(self, model, dataset, device):
        accuracy, latency = evaluate(model, dataset, device)
        return accuracy, latency

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint, first=False):
        while True:
            # config = {'d': np.random.choice[0, 1], 'e': np.random.choice(self.expand_list)}
            config = None
            if first:
                config = self.supernet.mutate_sample(self.supernet.min_subnet_arch(), 0.4)
                self.model.set_active_subnet(**config)
            else:
                config = self.model.sample_active_subnet()
            if 'w' in config:
                del config['w']
            accuracy, efficiency = self.accuracy_efficiency_predictor(self.model, self.dataset, self.device)
            print(accuracy, efficiency)
            if efficiency <= constraint:
                return config, accuracy, efficiency

    def mutate_sample(self, sample, constraint):
        while True:
            mutated_sample = self.supernet.mutate_sample(sample_arch=sample, mut_prob=self.arch_mutate_prob)
            self.model.set_active_subnet(**mutated_sample)
            accuracy, efficiency = self.accuracy_efficiency_predictor(self.model, self.dataset, self.device)
            print(f"Mutate: {self.arch_mutate_prob} {accuracy} {efficiency}")

            if efficiency <= constraint:
                self.arch_mutate_prob = self.init_arch_mutate_prob
                return mutated_sample, accuracy, efficiency
            else:
                self.arch_mutate_prob = min(2 * self.arch_mutate_prob, 0.9)

    def crossover_sample(self, sample1, sample2, constraint):
        counter = 0
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                else:
                    for i in range(len(new_sample[key])):
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )

            self.model.set_active_subnet(**new_sample)
            accuracy, efficiency = self.accuracy_efficiency_predictor(self.model, self.dataset, self.device)
            if efficiency <= constraint:
                return new_sample, accuracy, efficiency
            else:
                counter += 1
                if counter > 100:
                    self.model.set_active_subnet(**sample2)
                    accuracy, efficiency = self.accuracy_efficiency_predictor(self.model, self.dataset, self.device)
                    return sample2, accuracy, efficiency


    def run_evolution_search(self, constraint, verbose=False, first=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        best_info = None
        if verbose:
            print("Generate random population...")
        if first:
            accuracy, efficiency = self.accuracy_efficiency_predictor(self.model, self.dataset, self.device)

        for _ in tqdm(range(self.population_size)):
            sample, accuracy, latency = self.random_valid_sample(constraint, first=first)
            population.append((accuracy, sample, latency))

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        with tqdm(
            total=self.max_time_budget,
            desc="Searching with constraint (%s)" % constraint,
            disable=False,
        ) as t:
            for i in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
                acc = parents[0][0]
                t.set_postfix({"acc": parents[0][0]})
                if verbose and (i + 1) % 100 == 0:
                    print("Iter: {} Acc: {}".format(i + 1, parents[0][0]))

                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents

                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    new_sample, accuracy, efficiency = self.mutate_sample(par_sample, constraint)
                    population.append((accuracy, new_sample, efficiency))

                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]

                    new_sample, accuracy, efficiency = self.crossover_sample(
                        par_sample1, par_sample2, constraint
                    )
                    population.append((accuracy, new_sample, efficiency))
                t.update(1)

        return best_valids, best_info
