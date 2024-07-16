
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_nature_and_hoperator(NatureDataPath,NPlot):
    Data = np.load(NatureDataPath,allow_pickle=True)
    XNature=Data['XNature']
    YObs=Data['YObs']
    ObsLoc=Data['ObsLoc']
    if not isinstance(NPlot, int):
        NPlot = 200  #Plot the last NPlot times.
    #Plot the observations
    tmpnobs = XNature.shape[0]
    tmpntimes = XNature.shape[2]
    tmpobs = np.reshape(YObs[:,0], [tmpntimes, tmpnobs]).transpose()

    xobs = np.reshape(ObsLoc[:,0], [tmpntimes, tmpnobs]).transpose()
    tobs = np.reshape(ObsLoc[:,1], [tmpntimes, tmpnobs]).transpose()

    fig , axs = plt.subplots( 1 , 2 , figsize=(12,5),sharey=True)

    clevs1 = np.arange(-16, 16.1, 0.5)
    clevs2 = np.arange(0.0, 60.1, 1.0)
    cmap1=axs[0].contourf(XNature[:,0,-NPlot:],clevs1,cmap='RdBu_r')
    axs[0].set_ylabel('Observation location')
    axs[0].set_xlabel('Time')
    axs[0].set_title('(a)')
    cbar1=fig.colorbar(cmap1,ax=axs[0])
    cbar1.set_label('X')

    cmap2=axs[1].contourf(tmpobs[:,-NPlot:],clevs2,cmap='gist_ncar')
    cbar2=fig.colorbar(cmap2,ax=axs[1])
    cbar2.set_label('Reflectivity')
    axs[1].set_xlabel('Time')
    axs[1].set_title('(b)')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig( './FigureNatureReflectivity.png', facecolor='w', format='png' )
    plt.show(block=False)
    plt.close()




