# -*- coding: utf-8 -*-
'''
Main script to run a simulation
'''
import conf

def create_experiment(config):
    a=0
def run_nature(config):
    a=0
def run_da(config):
    a=0
def run_plots(config):
    a=0

if __name__ == "__main__":
    ###########################################################################
    #------------------------ Load the configurations ------------------------#
    ###########################################################################
    GeneralConf = conf.GeneralConf
    DAConf      = conf.DAConf
    ModelConf   = conf.ModelConf
    PlotConf    = conf.ModelConf
    
    if GeneralConf['NewExperiment']:
        create_experiment(GeneralConf)
    if GeneralConf['RunNature']:
        run_nature(ModelConf)
    if GeneralConf['RunDA']:
        run_da(DAConf)
    if GeneralConf['RunPlots']:
        run_plots(PlotConf)
        