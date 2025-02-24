"""

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import json
import joblib
import numpy as np
import torch
import os.path as osp
import time
import atexit
import os
import warnings
from datetime import datetime
from utils.stat_tools import statistics_scalar
from utils.serialization_utils import convert_json

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return f'\x1b[{";".join(attr)}m{string}\x1b[0m'


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        
        # Get the absolute path to the directory where main.py is located
        try:
            base_dir = osp.dirname(osp.abspath(__file__))
            # Move up one directory to the parent directory (same level as main.py)
            parent_dir = osp.dirname(base_dir)
        except NameError:
            base_dir = osp.abspath(osp.curdir)
            parent_dir = base_dir

        # Construct the output directory path relative to the location of main.py
        if not output_dir:
            # Using a more readable timestamp format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if exp_name == None:
                dir_name = f"output/{timestamp}"
            else:
                dir_name = f"output/{exp_name}/{timestamp}"
            self.output_dir = osp.join(parent_dir, dir_name)
        else:
            # Handle absolute and relative paths appropriately
            if osp.isabs(output_dir):
                self.output_dir = output_dir
            else:
                self.output_dir = osp.join(parent_dir, output_dir)
        
        if osp.exists(self.output_dir):
            print(f"Warning: Log dir {self.output_dir} already exists! Storing info there anyway.")
        else:
            os.makedirs(self.output_dir)

        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize(f"Logging data to {self.output_file.name}", 'green', bold=True))
    
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):

        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def save_config(self, config):

        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        # print(output)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)

    def save_state(self, state_dict, itr=None):

        
        fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, fname))
        except:
            self.log('Warning: could not pickle state_dict.', color='red')
        if hasattr(self, 'pytorch_saver_elements'):
            self._pytorch_simple_save(itr)

    

    def setup_pytorch_saver(self, what_to_save):

        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        
        assert hasattr(self, 'pytorch_saver_elements'), \
            "First have to setup saving with self.setup_pytorch_saver"
        fpath = 'pyt_save'
        fpath = osp.join(self.output_dir, fpath)
        fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            torch.save(self.pytorch_saver_elements, fname)


    def dump_tabular_leg(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
    
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else str(val)
            print(f"| {key:<{max_key_len}} | {valstr:>15} |")
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return statistics_scalar(vals)
