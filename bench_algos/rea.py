import argparse
import collections
from copy import deepcopy
import logging
import os
import sys
import random

import pickle

from nashpobench2api import NASHPOBench2API as API
from nashpobench2api.utils import get_ith_op, set_ith_op


def main(args):
    # make save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # make the API instance
    api = API(data_dir='../data', verbose=False)

    # run the algorithm args.runs times
    for i in range(args.runs):
        # set seed
        random.seed(i)
        # run
        results = regularized_evolution(args, api)
        # reset log
        api.reset_logdata()
        # save results
        save_path =os.path.join( args.save_dir, f'seed{i}.pkl') 
        with open( save_path, 'wb' ) as f:
            pickle.dump(results, f)
 

class Model:
    def __init__(self,key,acc):
        self.key = key
        self.acc = acc


def mutate(key: dict, api):
    new_key = deepcopy(key)
    mutate_elements = [f'edge-{i}' for i in range(6)] + ['lr'] + ['batch_size']
    e = random.choice(mutate_elements) 

    if e.startswith('edge'):
        i = int( e[-1] )
        cur_op = get_ith_op( key['cellcode'], i)
        candidates = [i for i in range(4) if i != cur_op ]
        new_op = random.choice( candidates )
        new_key['cellcode'] = set_ith_op(key['cellcode'], i, new_op)
    elif e == 'lr':
        candidates = [l for l in api.lrs if l != key['lr'] ]
        new_key['lr'] = random.choice( candidates )
    elif e == 'batch_size':
        candidates = [b for b in api.batch_sizes if b != key['batch_size'] ]
        new_key['batch_size'] = random.choice( candidates )

    return new_key


#Algorithm for regularized evolution (i.e. aging evolution).
def regularized_evolution(args, api):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".
    """
    # init rea objects
    population = collections.deque()
    sample = []

    # init population
    while len(population) < args.population_size:
        key = api.get_random_key()
        # query accuracy
        acc, cost = api.query_by_key(**key, epoch=12 if args.use_proxy else 200)

        # append the info
        population.append(Model(key,acc))

    # Carry out evolution in cycles. Each cycle produces a model and removes another.
    while api.get_total_cost() < args.time_budget:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < args.sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.acc)

        # Create the child model and store it.
        child_key = mutate(parent.key, api)
        # query accuracy
        acc, cost = api.query_by_key(**child_key, epoch=12 if args.use_proxy else 200)
        # Append the info
        population.append(Model(child_key, acc))

        # Remove the oldest model.
        population.popleft()

    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("REA")
    
    # hyperparameters for REA
    parser.add_argument("--population_size", type=int, default=10, help="The population size in REA.")
    parser.add_argument("--sample_size", type=int, default=3, help="The sample size in REA.")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--use_proxy",
        type=int,
        default=1,
        help="Whether to use the proxy (H0) task or not.",
    )
    #
    parser.add_argument(
        "--runs", type=int, default=500, help="The total runs for evaluation."
    )
    parser.add_argument(
        "--test_epoch",
        default='both',
        help='The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/rea",
        help="Folder to save checkpoints and log.",
    )
    args = parser.parse_args()

    main(args)
