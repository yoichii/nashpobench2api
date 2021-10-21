import argparse
import logging
import os
import sys
import time
import random

import pickle

from nashpobench2api import NASHPOBench2API as API


def main(args):
    # make the API instance
    api = API('../data')

    # run random saerch args.runs times
    print('>> random search')
    for i in range(args.runs):
        args.rand_seed = i
        results = run(args, api)
        api.reset_logdata()
    print()

    # len
    print('>> len')
    print( len(api) )
    print()

    # str
    print('>> print')
    print(api)
    print()

    # loop
    api.reset_logdata(verbose=False)
    print('>> loop')
    for i in api:
        pass
    print()

    # idx to key
    print('>> idx to key')
    for i in range(10):
        print( api.get_key_from_idx(random.choice(range(len(api)))) )
    print()

    # key to idx
    print('>> key to idx')
    for i in range(10):
        print( api.get_idx_from_key(api.get_random_key()) ) 
    print()

    # query by index
    print('>> query by idx')
    for i in range(10):
        print( api.query_by_index(
            random.choice(range(len(api.cellcodes))),
            random.choice(range(len(api.lrs))),
            random.choice(range(len(api.batch_sizes)))
        ))
    print()
        
    # query the data in the middle of the train
    print('>> query by idx in the middle of the train')
    key = api.get_random_key()
    for mode in ['train','valid','test']:
        for iepoch in [1,3,9]:
            api.query_by_key(**key, mode=mode, epoch=12, iepoch=iepoch)
    print()

    # get results
    print('>> get results')
    for epoch in [12, 200, 'both']:
        print(api.get_results(epoch=epoch)['final_accs'])
    print()


# random search
def run(args, api):
    while api.get_total_cost() < args.time_budget:
        # get a random key (=cellcode, lr, batch_size)
        key = api.get_random_key()
        # query data
        acc, cost = api.query_by_key(**key, epoch=12)
    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="The total runs for evaluation."
    )
    parser.add_argument(
        "--test_epoch",
        default='both',
        help='The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).',
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="manual seed")
    args = parser.parse_args()

    main(args)
