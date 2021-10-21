import argparse
import os
import random
import sys

import pickle

from nashpobench2api import NASHPOBench2API as API
from rea_nas import do_nas
from random_search_hpo import do_hpo


def main(args):
    # make save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # make the API instance
    api = API(data_dir='../data', verbose=False)

    # run the algorithm args.runs times
    for i in range(args.runs):
        # set seed
        args.rand_seed = i
        random.seed(args.rand_seed)
        # run
        results = run(args, api)
        # reset log
        api.reset_logdata()
        # save results
        save_path =os.path.join( args.save_dir, f'seed{i}.pkl') 
        with open( save_path, 'wb' ) as f:
            pickle.dump(results, f)


def run(args, api):
    # hpo with random cell
    cellcode = random.choice(api.cellcodes)
    hpo_results = do_hpo(args, api, cellcode) 
    # nas with fixed hp
    lr = hpo_results['best_key_trans'][-1]['lr']
    batch_size = hpo_results['best_key_trans'][-1]['batch_size']
    assert cellcode == hpo_results['best_key_trans'][-1]['cellcode']        
    return do_nas(args, api, lr, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "RS then REA"
    )
    # COMMON
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
        default="logs/rs_then_rea",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="manual seed")
    # RS
    parser.add_argument(
        "--time_budget_hpo",
        type=int,
        default=5000,
        help="The total time cost budge for searching (in seconds).",
    )
    
    # REA
    parser.add_argument("--population_size", type=int, default=10, help="The population size in EA.")
    parser.add_argument("--sample_size", type=int, default=3, help="The sample size in EA.")
    parser.add_argument(
        "--use_proxy",
        type=int,
        default=1,
        help="Whether to use the proxy (H0) task or not.",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    
    args = parser.parse_args()

    main(args)
