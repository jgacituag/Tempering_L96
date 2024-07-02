#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:59:53 2023

@author: jgacitua
"""

if NatureConf['RunPlot']:
    NPlot = 1000  # Plot the last NPlot times.
    print('Ploting the output')
    start = time.time()
    FigPath = GeneralConf['FigPath']
    ExpName = GeneralConf['ExpName']
    if not os.path.exists(FigPath + '/' + ExpName):
        os.makedirs(FigPath + '/' + ExpName)

    # Plot the observations
    tmpnobs = int(np.arange(1, Nx + 1, int(1 / ObsConf['SpaceDensity'])).size)
    tmpntimes = int(np.shape(XNature)[2])
    tmpobs = np.reshape(YObs[:, 0], [tmpntimes, tmpnobs]).transpose()
    xobs = np.reshape(ObsLoc[:, 0], [tmpntimes, tmpnobs]).transpose()
    tobs = np.reshape(ObsLoc[:, 1], [tmpntimes, tmpnobs]).transpose()

    # Plot the nature run
    plt.figure()
    plt.pcolor(XNature[:, 0, -NPlot:], vmin=-15, vmax=15, cmap='RdBu_r')
    plt.xlabel('Time')
    plt.ylabel('Grid points')
    plt.title('X True')
    plt.colorbar()
    plt.savefig(FigPath + '/' + ExpName + '/Nature_run_X.png', facecolor='w', format='png')
    plt.show(block=False)

    plt.figure()
    plt.pcolor(tmpobs[:, -NPlot:], vmin=-15, vmax=15, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Observation location')
    plt.title('Observations')
    plt.savefig(FigPath + '/' + ExpName + '/Nature_run_Y.png', facecolor='w', format='png')
    plt.show(block=False)