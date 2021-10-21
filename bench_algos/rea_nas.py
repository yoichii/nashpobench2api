import argparse
import collections
from copy import deepcopy
import logging
import os
import sys
import random

import pickle

from nashpobench2api import NASHPOBench2API
from nashpobench2api.utils import get_ith_op, set_ith_op


class Model:
    def __init__(self,key,acc):
        self.key = key
        self.acc = acc


def mutate(key: dict):
    new_key = deepcopy(key)
    mutate_elements = [f'edge-{i}' for i in range(6)]
    e = random.choice(mutate_elements) 

    # mutate cell (lr, batch are fixed)
    i = int( e[-1] )
    cur_op = get_ith_op( key['cellcode'], i)
    candidates = [i for i in range(4) if i != cur_op ]
    new_op = random.choice( candidates )
    new_key['cellcode'] = set_ith_op(key['cellcode'], i, new_op)

    return new_key


#Algorithm for regularized evolution (i.e. aging evolution).
def regularized_evolution(
    api,
    lr,
    batch_size,
    args
):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".
    """

    # init rea objects
    population = collections.deque()
    sample = []

    # init population
    while len(population) < args.population_size:
        # a random setting with fixed hp
        key = api.get_random_key()
        key['lr'] = lr
        key['batch_size'] = batch_size
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
        child_key = mutate(parent.key)
        # query accuracy
        acc, cost = api.query_by_key(**child_key, epoch=12 if args.use_proxy else 200)
        # Append the info
        population.append(Model(child_key, acc))

        # Remove the oldest model.
        population.popleft()

    return api.get_results(epoch=args.test_epoch)


def do_nas(args, api, lr, batch_size):
    # set seed
    random.seed(args.rand_seed)
 
    results = regularized_evolution(
        api,
        lr,
        batch_size,
        args
    )
    return results
