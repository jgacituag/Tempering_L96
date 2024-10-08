#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
sys.path.append("/media/jgacitua/Backup/Tempering_L96/Plots_routines/")

def run_plot_process(conf):
    GeneralConf = conf.GeneralConf
    DAConf = conf.DAConf
    ModelConf = conf.ModelConf
    ObsConf = conf.ObsConf
    NatureConf = conf.NatureConf
    FigPath = GeneralConf['FiguresPath']
    os.makedirs(FigPath, exist_ok=True)
    
    if NatureConf['RunPlot']:
        naturefile=os.path.join(GeneralConf['DataPath'], NatureConf['NatureFileName'])
        NPlot = NatureConf['NPlot']
        if NatureConf['RunPlotNatureHoperator']:
            from Plot_Nature_and_Reflectivity import plot_nature_and_hoperator
            plot_nature_and_hoperator(naturefile,FigPath, NPlot)

        if NatureConf['RunPlotNatureObsGIF']:
            from Plot_Nature_and_Obs_TS import plot_nature_and_obs
            plot_nature_and_obs(naturefile,ModelConf,GeneralConf,ObsConf,NPlot)
            
    #if config['RunPlotState']:
    #    plot_state_estimation(config)
    #if config['RunPlotForcing']:
    #    plot_forcing_estimation(config)
    #if config['RunPlotParameters']:
    #    plot_parameter_estimation(config)