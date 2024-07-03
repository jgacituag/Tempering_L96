# -*- coding: utf-8 -*-
'''
General Funtions to handle directories, logs and configurations
'''

import os
import shutil
import logging
import itertools
import json

def generate_combinations(config):
    keys, values = zip(*[(k, v) for k, v in config.ModelConf.items() if isinstance(v, (list, np.ndarray))])
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    for i, combo in enumerate(combinations):
        combo['name'] = f"{i:03d}"
    logging.info("Generated %d combinations", len(combinations))
    return combinations

def save_configurations(combinations, base_path):
    combo_file = os.path.join(base_path, "combinations.csv")
    with open(combo_file, 'w') as f:
        for combo in combinations:
            f.write(f"{combo['name']},{json.dumps(combo)}\n")
    logging.info("Saved configuration combinations to %s", combo_file)


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
