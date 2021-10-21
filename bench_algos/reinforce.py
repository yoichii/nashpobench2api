import argparse
from copy import deepcopy
import logging
import os
import sys
import random

import numpy as np
import pickle
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

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
        args.rand_seed = i
        # run
        results = reinforce(args, api)
        # reset log
        api.reset_logdata()
        # save results
        save_path =os.path.join( args.save_dir, f'seed{i}.pkl') 
        with open( save_path, 'wb' ) as f:
            pickle.dump(results, f)


class PolicyTopology(nn.Module):
    def __init__(
            self,
            num_edges: int,
            lrs: list,
            batch_sizes: list
        ):
        super(PolicyTopology, self).__init__()
        # search space
        self.num_edges = num_edges
        self.ops = [0,1,2,3]
        self.lrs = deepcopy( lrs )
        self.batch_sizes = deepcopy( batch_sizes )
        # params
        self.cell_params = nn.Parameter(
            1e-3 * torch.randn(self.num_edges, len(self.ops))
        )
        self.lr_params = nn.Parameter(
            1e-3 * torch.randn(1, len(self.lrs))
        )
        self.batch_params = nn.Parameter(
            1e-3 * torch.randn(1, len(self.batch_sizes))
        )

    def generate_key(self, actions):
        # cellcode
        cellcode = ''
        for i in range(self.num_edges):
            cellcode += str( actions[i] )
            if i == 0 or i == 2:
                cellcode += '|'
        # lr
        lr = self.lrs[ actions[self.num_edges] ]
        # batch size
        batch_size = self.batch_sizes[ actions[self.num_edges+1] ]
        # return key
        return {
            'cellcode': cellcode,
            'lr': lr,
            'batch_size': batch_size
        }

    def get_key(self):
        with torch.no_grad():
            # cellcode
            cellcode = ''
            for i in range(self.num_edges):
                op_weights = self.cell_params[i]
                op = self.ops[ op_weights.argmax().item() ]
                cellcode += str(op)
                if i == 0 or i == 2:
                    cellcode += '|'
            # lr
            lr = self.lrs[ self.lr_params.argmax().item() ]
            # batch size
            batch_size = self.batch_sizes[ self.batch_params.argmax().item() ]
        # return key
        return { 'cellcode': cellcode,
            'lr': lr,
            'batch_size': batch_size
        }

    def forward(self):
        #all_params = torch.cat(
        cell_prob = F.softmax(self.cell_params, dim=-1)
        lr_prob = F.softmax(self.lr_params, dim=-1)
        batch_prob = F.softmax(self.batch_params, dim=-1)
        return cell_prob, lr_prob, batch_prob


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


def select_action(policy):
    # prob
    cell_prob, lr_prob, batch_prob = policy()
    # distributions
    cell_d = Categorical(cell_prob)
    lr_d = Categorical(lr_prob)
    batch_d = Categorical(batch_prob)
    # actions
    cell_action = cell_d.sample()
    lr_action = lr_d.sample()
    batch_action = batch_d.sample()
    actions = torch.cat( (cell_action, lr_action, batch_action))
    # log_prob
    cell_log = cell_d.log_prob(cell_action)
    lr_log =  lr_d.log_prob(lr_action)
    batch_log =  batch_d.log_prob(batch_action)
    log_probs = torch.cat( (cell_log, lr_log, batch_log) )
    return log_probs, actions.cpu().tolist()


def reinforce(args, api):
    random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PolicyTopology(
            6,
            api.lrs,
            api.batch_sizes
        ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    eps = np.finfo(np.float32).eps.item()
    baseline = ExponentialMovingAverage(args.EMA_momentum)
    total_steps = 0
    logging.info("policy    : {:}".format(policy))
    logging.info("optimizer : {:}".format(optimizer))
    logging.info("eps       : {:}".format(eps))

    # REINFORCE
    while api.get_total_cost() < args.time_budget:
        log_prob, action = select_action(policy)
        key = policy.generate_key(action)
        # query accuracy
        reward, cost = api.query_by_key(**key, epoch=12)
        # calculate loss
        baseline.update(reward)
        policy_loss = (-log_prob * (reward - baseline.value())).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        # accumulate time
        total_steps += 1
        logging.info(
            "step [{:3d}] : average-reward={:.3f} : policy_loss={:.4f} : {:}".format(
                total_steps, baseline.value(), policy_loss.item(), policy.get_key()
            )
        )

    return api.get_results(epoch=args.test_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("REINFORCE")
    parser.add_argument(
        "--learning_rate", default=0.01, type=float, help="The learning rate for REINFORCE."
    )
    parser.add_argument(
        "--EMA_momentum", type=float, default=0.9, help="The momentum value for EMA."
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
    parser.add_argument(
        "--test_epoch",
        default='both',
        help='The test epoch. 12 (trained), 200 (suggorage), or both (12 and 200).',
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default='logs/reinforce',
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="manual seed")
    args = parser.parse_args()

    main(args)
