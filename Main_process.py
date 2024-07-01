# -*- coding: utf-8 -*-
import os
import logging
from directory_manager import create_and_change_directory
from config_manager import generate_combinations, save_configurations
from experiment_process import run_nature_process, run_da_process, run_plot_process
from logger import setup_logger


import config

def main():
    setup_logger(config.GeneralConf['ExpPath'])
    logging.info("Starting experiment: %s", config.GeneralConf['ExpName'])
    
    # Create or change to experiment directory
    if config.GeneralConf['NewExperiment']:
        create_and_change_directory(config.GeneralConf['ExpPath'], reset=True)
    else:
        create_and_change_directory(config.GeneralConf['ExpPath'], reset=False)

    if config.GeneralConf.get('MultipleConfigurations', False):
        handle_multiple_configurations()
    else:
        handle_single_configuration()

def handle_multiple_configurations():
    combinations = generate_combinations(config)
    save_configurations(combinations, config.GeneralConf['ExpPath'])
    
    for combo in combinations:
        exp_path = os.path.join(config.GeneralConf['ExpPath'], combo['name'])
        create_and_change_directory(exp_path)
        run_single_experiment(combo)

def handle_single_configuration():
    run_single_experiment(config)

def run_single_experiment(exp_config):
    if exp_config.GeneralConf['RunNature']:
        run_nature_process(exp_config)
    if exp_config.GeneralConf['RunDA']:
        run_da_process(exp_config)
    if exp_config.GeneralConf['RunPlots']:
        run_plot_process(exp_config)

if __name__ == "__main__":
    main()