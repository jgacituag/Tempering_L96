# -*- coding: utf-8 -*-
'''
General Functions to handle directories, logs, and configurations
'''
import numpy as np
import os
import shutil
import logging
import itertools
import json

def generate_combinations(config):
    multi_vars = config.GeneralConf['MultivalueVariables']
    all_values = [getattr(config, section)[var] for section, var in multi_vars]
    keys = [f"{section}_{var}" for section, var in multi_vars]

    combinations = []
    for i, combo_values in enumerate(itertools.product(*all_values)):
        combo = dict(zip(keys, combo_values))
        combo_name = f"{i:03d}"
        combo['name'] = combo_name
        combo['GeneralConf_DataPath'] = os.path.join(config.GeneralConf['DataPath'], combo_name)
        combinations.append(combo)

    logging.info("Generated %d combinations", len(combinations))
    return combinations

def save_configurations(combinations, base_path):
    combo_file = os.path.join(base_path, "combinations.csv")
    with open(combo_file, 'w') as f:
        for combo in combinations:
            f.write(f"{combo['name']},{json.dumps(convert_to_serializable(combo))}\n")
    logging.info("Saved configuration combinations to %s", combo_file)

def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)
    else:
        return data

def create_and_change_directory(path, reset=False):
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    logging.info("Changed directory to %s", path)

def setup_logger(exp_path):
    log_file = os.path.join(exp_path, 'experiment.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
