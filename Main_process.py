# -*- coding: utf-8 -*-
import os
import logging
import experiment_config as config
from GeneralFuntions import setup_logger
from GeneralFuntions import create_and_change_directory
from GeneralFuntions import generate_combinations
from GeneralFuntions import save_configurations
from Run_Nature_Process import run_nature_process
from Run_DA_Process import run_da_process
from Run_Plot_Process import run_plot_process
import sys
sys.path.append(f"{config['GeneralConf']['FortranRoutinesPath']}/model/")
sys.path.append(f"{config['GeneralConf']['FortranRoutinesPath']}/data_assimilation/")

def main():
    '''
    main function
    '''
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
    '''
    For Multiple configurations create subdirectories 
    where to run a experiment with a single configuration 
    '''
    combinations = generate_combinations(config)
    save_configurations(combinations, config.GeneralConf['ExpPath'])
    for combo in combinations:
        exp_path = os.path.join(config.GeneralConf['ExpPath'], combo['name'])
        create_and_change_directory(exp_path)
        run_single_experiment(combo)

def handle_single_configuration():
    '''
    General funtion to run a experiment with a single configuration 
    '''
    run_single_experiment(config)

def run_single_experiment(exp_config):
    '''
    General funtion to run a experiment with a single configuration 
    '''
    if exp_config.GeneralConf['RunNature']:

        run_nature_process(exp_config)
    if exp_config.GeneralConf['RunDA']:

        run_da_process(exp_config)
    if exp_config.GeneralConf['RunPlots']:

        run_plot_process(exp_config)

if __name__ == "__main__":
    main()
