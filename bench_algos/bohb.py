import argparse
import collections
import logging
import numpy as np
import os
import random
import sys
import time
from copy import deepcopy

# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
import pickle

from nashpobench2api.api import NASHPOBench2API as API


def main(args):
    # make save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # make the API instance
    logging.basicConfig(level=logging.ERROR)
    api = API(data_dir='../data', verbose=False)

    # run the algorithm args.runs times
    for i in range(args.runs):
        # set seed
        args.rand_seed = i
        # run
        results = bohb(args, api)
        # reset log
        api.reset_logdata()
        # save results
        save_path =os.path.join( args.save_dir, f'seed{i}.pkl') 
        with open( save_path, 'wb' ) as f:
            pickle.dump(results, f)


def get_config_space(lrs, batch_sizes, rand_seed):
    cs = ConfigSpace.ConfigurationSpace(rand_seed)
    # cellcode
    for i in range(6):
        cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter(str(i), [0,1,2,3])
            )
    # lr
    cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter('lr', lrs)
        )
    # batch size
    cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter('batch_size', batch_sizes)
        )
    return cs


def config2key(config):
    # cellcode
    cellcode = ''
    for i in range(6):
        cellcode += str( config[str(i)] )
        if i == 0 or i == 2:
            cellcode += '|'
    # lr
    lr = config['lr']
    # batch size
    batch_size = config['batch_size']
    # return key
    return {
        'cellcode': cellcode,
        'lr': lr,
        'batch_size': batch_size
    }


class MyWorker(Worker):
    def __init__(
            self,
            *args,
            config2key,
            api=None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.config2key = config2key
        self.api = api

    def compute(
            self,
            config,
            budget,
            **kwargs
        ):
        key = self.config2key(config)
        # query accuracy
        acc, cost = self.api.query_by_key(**key, epoch=12, iepoch=int(budget))
        return {"loss": 100 - acc, "info": key}


def bohb(args, api):
    # set seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    # get config space
    cs = get_config_space(api.lrs, api.batch_sizes, args.rand_seed)

    hb_run_id = "0"
    # execute
    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            config2key=config2key,
            api=api,
            run_id=hb_run_id,
            id=i,
        )
        w.run(background=True)
        workers.append(w)

    bohb_inst = BOHB(
        configspace=cs,
        run_id=hb_run_id,
        eta=3,
        min_budget=1,
        max_budget=12,
        nameserver=ns_host,
        nameserver_port=ns_port,
        num_samples=args.num_samples,
        random_fraction=args.random_fraction,
        bandwidth_factor=args.bandwidth_factor,
        ping_interval=10,
        min_bandwidth=args.min_bandwidth,
    )

    bohb_inst.run(args.n_iters, min_n_workers=num_workers)

    bohb_inst.shutdown(shutdown_workers=True)
    NS.shutdown()

    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BOHB"
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--runs", type=int, default=500, help="The total runs for evaluation."
    )
    # BOHB
    parser.add_argument(
        "--strategy",
        default="sampling",
        type=str,
        nargs="?",
        help="optimization strategy for the acquisition function",
    )
    parser.add_argument(
        "--min_bandwidth",
        default=0.3,
        type=float,
        nargs="?",
        help="minimum bandwidth for KDE",
    )
    parser.add_argument(
        "--num_samples",
        default=4,
        type=int,
        nargs="?",
        help="number of samples for the acquisition function",
    )
    parser.add_argument(
        "--random_fraction",
        default=0.0,
        type=float,
        nargs="?",
        help="fraction of random configurations",
    )
    parser.add_argument(
        "--bandwidth_factor",
        default=3,
        type=int,
        nargs="?",
        help="factor multiplied to the bandwidth",
    )
    parser.add_argument(
        "--n_iters",
        default=40,
        type=int,
        nargs="?",
        help="number of iterations for optimization method",
    )
    parser.add_argument(
        "--test_epoch",
        default='both',
        help='The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/bohb",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="manual seed")
    args = parser.parse_args()

    main(args)
