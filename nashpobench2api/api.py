from itertools import product
import logging
from logging import Logger
import os
import random
import sys
from typing import Union, Optional

import pickle

class NASHPOBench2API:

    def __init__(
            self,
            data_dir: str = 'data',
            seed: Optional[int] = None,
            logger: Optional[Logger] = None,
            verbose: bool = True
        ):
        self.data_dir = data_dir
        self.logger = logger
        self.verbose = verbose
        if logger is None:
            self.logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)
            self.logger.propagate = False
        try:
            for p in [ 'bench12.pkl', 'cellinfo.pkl', 'avgaccs200.pkl']:
                data_path = os.path.join(data_dir, p)
                with open(data_path, 'rb') as f:
                    exec(f'self.{p.split(".")[0]} = pickle.load(f)')
        except Exception as e:
            self.logger.error('HINT: Download the API data from Google Drive, and set the path like NASHPOBench2API("path/to/data")')
            raise e 
            return
            
        # set serach space
        self.cellcodes = sorted(list(set(self.cellinfo['hash'].keys()))) 
        self.lrs  = sorted( list(set(self.bench12['lr'].values())) )
        self.batch_sizes  = sorted( list(set(self.bench12['batch_size'].values())) )
        self.seeds = sorted( list(set(self.bench12['seed'].values())) )
        self.epochs = [12, 200]
        # set objects for log
        self._init_logdata()
        # set seed
        if seed:
            self.set_seed(seed)
        else:
            self.seed = self.seeds[0]

    def _init_logdata(self):
        ## acc
        self.acc_trans = []
        self.best_acc = -1
        self.best_acc_trans = []
        ## cost
        self.total_cost = 0.0
        self.total_cost_trans = []
        self.best_cost = None
        ## key
        self.key_trans = []
        self.best_key = None
        self.best_key_trans = []
        # set verbose
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        return

    def __len__(self):
        return len(self.cellcodes) * len(self.lrs) * len(self.batch_sizes)

    def __str__(self):
        return 'NAS-HPO-Bench-II'


    def get_key_from_idx(
            self,
            idx: int
        ):
        key = {
            'cellcode': self.cellcodes[int( idx / ( len(self.lrs)*len(self.batch_sizes) ) )],
            'lr': self.lrs[int( idx / len(self.batch_sizes) ) % len(self.lrs) ],
            'batch_size': self.batch_sizes[ idx % len(self.batch_sizes) ]
        }
        return key

    def get_idx_from_key(
            self,
            key: dict
        ):
        cellidx = self.cellcodes.index(key['cellcode'])
        lridx = self.lrs.index(key['lr'])
        batchidx = self.batch_sizes.index(key['batch_size'])
        return cellidx*len(self.lrs)*len(self.batch_sizes) + lridx*len(self.batch_sizes) + batchidx

    def __getitem__(
            self,
            idx: int,
        ):
        key = self.get_key_from_idx(idx) 
        return self.query_by_key(**key)
    
    def query_by_index(
            self,
            cell_idx: int,
            lr_idx: int,
            batch_size_idx: int,
            epoch: Union[int,str] = 12,
            iepoch: Optional[int] = None,
            seed: Optional[int] = None
        ):
        cellcode = self.cellcodes[cell_idx]
        lr = self.lrs[lr_idx]
        batch_size = self.batch_sizes[batch_size_idx]
        if seed:
            self.set_seed(seed)
        return self.query_by_key(cellcode, lr, batch_size, epoch=epoch, iepoch=iepoch)

    def query_by_key(
            self,
            cellcode: str,
            lr: float,
            batch_size: int,
            epoch: Union[int,str] = 12,
            mode: str = 'valid',
            iepoch: Optional[int] = None,
            seed: Optional[int] = None,
            enable_log: bool = True,
        ):
        # check if a key is valid
        self._check_key(cellcode, lr, batch_size, epoch)
        assert mode in ['train', 'valid', 'test'], ValueError(f'mode {mode} should be train, valid, or test')
        # iepoch
        if epoch != 12:
            assert iepoch is None, ValueError(f'iepoch is not available in epoch {epoch}')
        if iepoch == None:
            iepoch = epoch
        assert iepoch <= epoch, ValueError(f'iepoch {iepoch} is graeter than epoch {epoch}')
        # set seed
        if seed:
            self.set_seed(seed)
        # cellcode to hashvalue
        hashv = self._cellcode2hash(cellcode)


        # 12 epoch
        if epoch == 12:
            # acc
            bench = self.bench12
            acc = bench[f'{mode}_acc_{iepoch-1}'][(hashv, lr, batch_size, self.seed)]
            # cost
            ## if iepoch == 12 (, the cost is pre-calculated)
            if iepoch == epoch:
                if mode=='train':
                    cost = bench[f'total_train_time'][(hashv, lr, batch_size, self.seed)]
                elif mode=='valid':
                    cost = bench[f'total_trainval_time'][(hashv, lr, batch_size, self.seed)]
                elif mode=='test':
                    cost = bench[f'total_trainval_time'][(hashv, lr, batch_size, self.seed)] + \
                            bench[f'total_test_time'][(hashv, lr, batch_size, self.seed)]
            ## else (less than 12 epoch)
            else:
                if mode=='train':
                    time_modes = ['train']
                elif mode=='valid':
                    time_modes = ['train', 'valid']
                elif mode=='test':
                    time_modes = ['train', 'valid', 'test']
                tmp = [
                        bench[f'{m}_time_{i}'][(hashv, lr, batch_size, self.seed)]
                            for m,i in product(time_modes, range(iepoch)) 
                    ]
                cost = sum(tmp)

            key = {'cellcode': cellcode, 'lr': lr, 'batch_size': batch_size}
            if enable_log:
                self._write_log(acc, cost, key)
            return acc, cost

        # 200 epoch
        elif epoch == 200:
            # the expected value of test accuracy
            bench = self.avgaccs200
            acc = bench['avg_acc'][(hashv, lr, batch_size)]
            return acc, None

    def _write_log(
            self,
            acc: float,
            cost: float,
            key: dict,
        ):
        if len(self.acc_trans) == 0:
            self.logger.debug(
                    f'   {"valid acc":<8}   {"cost":<8}    {"cellcode"} {"lr":<7} {"batch_size":<3}'
            )
        self.logger.debug(
                f'{acc:>8.2f} %   {cost:>8.2f} sec  {key["cellcode"]} {key["lr"]:<7.5f} {key["batch_size"]:<3}'
        )
        # current status
        self.acc_trans.append(acc)
        self.key_trans.append(key)
        self.total_cost += cost
        self.total_cost_trans.append(self.total_cost)
        # update the best status
        if self.best_key is None or self.best_acc < acc:
            self.best_acc, self.best_cost, self.best_key = acc, cost, key
        # current best status
        self.best_acc_trans.append(self.best_acc)
        self.best_key_trans.append(self.best_key)
        return

    def get_total_cost(self):
        return self.total_cost
        
    def get_results(
            self,
            epoch: Union[int,str] = 'both',
            mode: str ='test'
        ):
        # log
        self.logger.info('-'*23+' finished '+'-'*23)
        self.logger.info('The best setting is') 
        self.logger.info(
                f'   {"valid acc":<8}   {"cost":<8}    {"cellcode"} {"lr":<7} {"batch_size":<3}'
        )
        self.logger.info(
            f'{self.best_acc:>8.2f} %   {self.best_cost:>8.2f} sec  {self.best_key["cellcode"]} {self.best_key["lr"]:<7.5f} {self.best_key["batch_size"]:<3}'
        )
        self.logger.info(f' in {len(self.key_trans)} trials ({self.total_cost:.2f} sec)')

        # get the test accuracies of the best-valid-acc model (finalaccs)
        if epoch == 'both':
            epochs = [12, 200]
        else:
            epochs = [epoch]
        self.logger.info('-' * 56)
        self.final_accs = []
        for e in epochs:
            final_acc, _ = self.query_by_key(**self.best_key, epoch=e, mode=mode, enable_log=False)
            self.final_accs.append(final_acc)
            self.logger.info(f'{e}-epoch {mode} accuracy is {final_acc:.2f}%')
        self.logger.info('-' * 56)
        # return results
        nlist = ['acc_trans', 'key_trans', 'best_acc_trans', 'best_key_trans', 'total_cost_trans', 'final_accs']
        return {n: eval('self.'+n, {'self': self}) for n in nlist}

    def reset_logdata(
            self,
            logger: Optional[Logger] = None,
            verbose: bool = None
        ):
        if logger is not None:
            self.logger = logger
        if verbose is not None:
            self.verbose = verbose
        self._init_logdata()
        return

    def _cellcode2hash(
            self,
            cellcode: str
        ):
        return self.cellinfo['hash'][cellcode]

    def get_random_key(self):
        cellcode = random.choice(self.cellcodes)
        lr = random.choice(self.lrs)
        batch_size = random.choice(self.batch_sizes)
        return {
                'cellcode':cellcode,
                'lr': lr,
                'batch_size': batch_size
            }

    def _check_key(
            self,
            cellcode: str,
            lr: float,
            batch_size: int,
            epoch: int
        ):
        if cellcode not in self.cellcodes:
            raise ValueError(f'choose a cellcode {cellcode} from search space.')
        if lr not in self.lrs:
            raise ValueError(f'choose lr from {self.lrs}.')
        if batch_size not in self.batch_sizes:
            raise ValueError(f'choose batch size from {self.batch_sizes}.')
        if epoch not in self.epochs:
            raise ValueError(f'choose epoch from {self.epochs}.')
        return

    def get_search_space(self):
        return self.cellcodes, self.lrs, self.batch_sizes

    def set_seed(self, seed: int):
        if seed in self.seeds:
            self.seed = seed
        else:
            raise ValueError(f'choose a seed value from {self.seeds}.')
