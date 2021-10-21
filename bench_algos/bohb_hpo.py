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


def get_config_space(cellcode, lrs, batch_sizes, args):
    cs = ConfigSpace.ConfigurationSpace(args.rand_seed)
    # cellcode
    cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter('cellcode', [cellcode])
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
    cellcode = config['cellcode']
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



def do_hpo(args, api, cellcode):
    # set seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    # get config space
    cs = get_config_space(cellcode, api.lrs, api.batch_sizes, args)

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

    bohb = BOHB(
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

    results = bohb.run(args.n_iters, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    return api.get_results(epoch=args.test_epoch)
