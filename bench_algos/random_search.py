import argparse
import os
import pickle
import random
import sys

from nashpobench2api import NASHPOBench2API as API


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
        results = random_search(args, api)
        # reset log
        api.reset_logdata()
        # save results
        save_path =os.path.join( args.save_dir, f'seed{i}.pkl') 
        with open( save_path, 'wb' ) as f:
            pickle.dump(results, f)
            

# Random Search for Hyper-Parameter Optimization, JMLR 2012
def random_search(args, api):
    while api.get_total_cost() < args.time_budget:
        # get keys(=cellcode, lr, batch_size)
        key = api.get_random_key()
        # query accuracy
        acc, cost = api.query_by_key(**key, epoch=12)
    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RANDOM")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--runs", type=int, default=500, help="The total runs for evaluation."
    )
    parser.add_argument(
        "--test_epoch",
        default='both',
        help='The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).',
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/random",
        help="Folder to save checkpoints and log.",
    )
    args = parser.parse_args()

    main(args)
