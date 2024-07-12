# -*- coding: utf-8 -*-
import os
import logging
import sys
import experiment_config as expconf
from GeneralFuntions import setup_logger
from GeneralFuntions import create_and_change_directory
from GeneralFuntions import generate_combinations
from GeneralFuntions import save_configurations
from Run_Nature_Process import run_nature_process
from Run_DA_Process import run_da_process
from Run_Plot_Process import run_plot_process

def main():
    '''
    main function
    '''

    # Create or change to experiment directory
    if expconf.GeneralConf['NewExperiment']:
        create_and_change_directory(expconf.GeneralConf['ExpPath'], reset=True)
    else:
        create_and_change_directory(expconf.GeneralConf['ExpPath'], reset=False)

    setup_logger(expconf.GeneralConf['ExpPath'])
    logging.info("Starting experiment: %s", expconf.GeneralConf['ExpName'])

    if expconf.GeneralConf.get('MultipleConfigurations', False):
        handle_multiple_configurations()
    else:
        handle_single_configuration()

def handle_multiple_configurations():
    '''
    For Multiple configurations create subdirectories 
    where to run an experiment with a single configuration 
    '''
    combinations = generate_combinations(expconf)
    save_configurations(combinations, expconf.GeneralConf['ExpPath'])
    for combo in combinations:
        run_single_experiment(expconf, combo)

def handle_single_configuration():
    '''
    General function to run an experiment with a single configuration 
    '''
    run_single_experiment(expconf)

def run_single_experiment(config, combo=None):
    '''
    General function to run an experiment with a single configuration 
    '''
    if combo:
        for key, value in combo.items():
            if '_' in key:
                section, var = key.split('_')
                getattr(config, section)[var] = value

    if config.GeneralConf['RunNature']:
        run_nature_process(config)
    if config.GeneralConf['RunDA']:
        run_da_process(config)
    if config.GeneralConf['RunPlots']:
        run_plot_process(config)

if __name__ == "__main__":
    main()
