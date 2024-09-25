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
from Run_Forecast_Process import run_forecast_process
from Run_Plot_Process import run_plot_process
import subprocess  # For running shell commands in Python

sys.path.append("/media/jgacitua/storage/Tempering_L96/Plots_routines/")

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
    For multiple configurations, create subdirectories 
    where an experiment with a single configuration can be run
    '''
    combinations = generate_combinations(expconf)
    save_configurations(combinations, expconf.GeneralConf['ExpPath'])
    
    for combo in combinations:
        run_single_experiment(expconf, combo)
    
    if expconf.GeneralConf.get('multi_queue_hydra', False):  # If multi_queue_hydra is enabled
        handle_multi_queue_hydra()

def handle_multi_queue_hydra():
    '''
    Handles execution of experiments across multiple nodes in the cluster
    '''
    base_path = os.getcwd()
    ncores = 12
    experiment_script = expconf.GeneralConf['ExperimentScript']
    nature_array = expconf.GeneralConf['NatureArray']  # Assuming this is a list of nature names
    
    for nature in nature_array:
        exp_num = os.getpid()  # Or generate a random number or unique identifier for the job script
        script_name = f"./tmp_script_{exp_num}.bash"
        log_path = f"./logs/{experiment_script}_{nature}.log"
        
        # Create the bash script dynamically
        with open(script_name, 'w') as bash_script:
            bash_script.write("#!/bin/bash\n")
            bash_script.write("source /opt/load-libs.sh 3\n")
            bash_script.write(f"cd {base_path}\n")
            bash_script.write('export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"\n')
            bash_script.write(f"export OMP_NUM_THREADS={ncores}\n")
            bash_script.write(f"python -u ./{experiment_script} {nature} > {log_path}\n")
        
        # Submit the job to the queue using qsub
        qsub_command = f"qsub -l nodes=1:ppn={ncores} {script_name}"
        logging.info(f"Submitting job for {nature} with command: {qsub_command}")
        subprocess.run(qsub_command, shell=True)
        
        # Sleep between submissions to prevent overloading the system
        subprocess.run("sleep 1", shell=True)

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
    #if config.GeneralConf['RunForecast']:
    #    run_forecast_process(config)
    if config.GeneralConf['RunPlots']:
        run_plot_process(config)
        
if __name__ == "__main__":
    main()
