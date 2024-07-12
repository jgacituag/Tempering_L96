import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:59:53 2023

@author: jgacitua
"""
#=================================================================
# PLOT THE NATURE RUN AND THE OBSERVATIONS : 
#=================================================================
def plot_nature_and_hoperator(NatureDataPath,NPlot):
    Data = np.load(NatureDataPath,allow_pickle=True)
    ObsConf=Data['ObsConf']
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

def plot_nature_and_obs(NatureDataPath,ModelConf,GeneralConf,ObsConf,NPlot):
   
    if not isinstance(NPlot, int):
        NPlot = 1000   #Plot the last NPlot times.


    Data = np.load(NatureDataPath,allow_pickle=True)
    #ObsConf = Data['ObsConf']
    XNature = Data['XNature']
    YObs = Data['YObs']
    ObsLoc = Data['ObsLoc']
    Nx = ModelConf['nx']
    FigPath = GeneralConf['FiguresPath']
    os.makedirs(FigPath, exist_ok=True)
    print('Ploting the output')
    start = time.time()

    #Plot the observations
    tmpnobs = int(np.arange(1, Nx+1, int(1/ObsConf['SpaceDensity']) ).size )
    tmpntimes = int(np.shape(XNature)[2] )
    tmpobs = np.reshape(YObs[:,0], [tmpntimes, tmpnobs]).transpose()

    xobs = np.reshape(ObsLoc[:,0], [tmpntimes, tmpnobs]).transpose()
    tobs = np.reshape(ObsLoc[:,1], [tmpntimes, tmpnobs]).transpose()

    #Plot the nature run.
    plt.figure()
    plt.pcolor(XNature[:,0,-NPlot:],vmin=-15,vmax=15,cmap='RdBu_r')
    plt.xlabel('Time')
    plt.ylabel('Grid points')
    plt.title('X True')
    plt.colorbar()
    plt.savefig( FigPath +'/Nature_run_X.png', facecolor='w', format='png' )
    plt.show(block=False)
    #plt.close()


    plt.figure()
    plt.pcolor(tmpobs[:,-NPlot:],vmin=-15,vmax=15,cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Observation location')
    plt.title('Observations')
    plt.savefig( FigPath + '/Nature_run_Y.png', facecolor='w', format='png' )
    plt.show(block=False)
    #plt.close()


    plt.figure()
    plt.plot(tmpobs[0,-NPlot:],'o',alpha=0.8)
    plt.plot(XNature[0,0,-NPlot:])
    plt.xlabel('Time')
    plt.ylabel('X at 1st grid point')
    plt.title('Nature and Observations')
    plt.savefig( FigPath + '/Nature_and_Obs_Time_Serie.png', facecolor='w', format='png' )
    plt.show(block=False)
    #plt.close()

    plt.figure()
    plt.plot(xobs[:,-1],tmpobs[:,-1],'o',alpha=0.8)
    plt.plot(np.arange(1,Nx+1),XNature[:,0,-1],'-o',alpha=0.8)
    plt.xlabel('Location')
    plt.ylabel('X at last time')
    plt.title('Nature and Observations')
    plt.savefig( FigPath + '/Nature_and_Obs_At_Last_Time.png', facecolor='w', format='png' )
    plt.show(block=False)
    #plt.close()



    def plot_for_image(data_in,obs_in,idx_obs_in,tobs_in):
            # Data for plotting
            data = np.append(data_in,data_in[0])
            obs  = np.append(obs_in,obs_in[0])
            idx_obs  = [int(i-1) for i in np.append(idx_obs_in,idx_obs_in[0])]
            data_size = np.shape(data)[0]
            theta = np.linspace(0,2*np.pi,data_size)#*np.pi/180
            fig, ax = plt.subplots(figsize=(8,8),subplot_kw={'projection': 'polar'})
            ax.plot(theta, data,label='Nature', color='IndianRed',linewidth=2.5)
            ax.plot(theta[idx_obs], obs,label='Obs', color='SteelBlue',linewidth=2.5)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(90)
            ax.legend(loc="upper left",bbox_to_anchor=(0.9, 1.05))
            ax.set_xticks(theta[:-1:1])
            ax.set_xticklabels([int(i) for i in np.arange(1,data_size-0.5,1)])
            
            ax.set_title(f'time {tobs_in:.3f}',fontsize=16)
            ax.set_rmax(20)
            ax.set_rmin(-20)
            ax.set_rticks(np.arange(-15,16,5))
            ax.grid(True)
        
            # Used to return the plot as an image rray
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image


    imageio.mimsave(FigPath + '/Nature_run_X.gif', [plot_for_image(XNature[:,0,i], tmpobs[:,i],xobs[:,i],ModelConf['dt']*np.unique(tobs)[i]) for i in range(NPlot)], fps=10)


    print('Ploting took ', time.time()-start, 'seconds.')


    plt.show()



def plot_data_assimilation_result(XAssim, YAssim, FigPath, ExpName):
    NPlot = 1000  # Plot the last NPlot times.
    
    # Plot the assimilated state
    plt.figure()
    plt.pcolor(XAssim[:, 0, -NPlot:], vmin=-15, vmax=15, cmap='RdBu_r')
    plt.xlabel('Time')
    plt.ylabel('Grid points')
    plt.title('Assimilated X')
    plt.colorbar()
    plt.savefig(FigPath + '/' + ExpName + '/Assimilated_X.png', facecolor='w', format='png')
    plt.show(block=False)

    # Plot the assimilated observations
    plt.figure()
    plt.pcolor(YAssim[:, -NPlot:], vmin=-15, vmax=15, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Observation location')
    plt.title('Assimilated Observations')
    plt.savefig(FigPath + '/' + ExpName + '/Assimilated_Y.png', facecolor='w', format='png')
    plt.show(block=False)

def run_plot_process2(conf):
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

def run_plot_process(conf):
    GeneralConf = conf.GeneralConf
    DAConf = conf.DAConf
    ModelConf = conf.ModelConf
    ObsConf = conf.ObsConf
    NatureConf = conf.NatureConf
    if NatureConf['RunPlot']:
        naturefile=os.path.join(GeneralConf['DataPath'], NatureConf['NatureFileName'])
        NPlot = NatureConf['NPlot']
        if NatureConf['RunPlotNatureHoperator']:
            plot_nature_and_hoperator(naturefile, NPlot)
        if NatureConf['RunPlotNatureObsGIF']:
            plot_nature_and_obs(naturefile,ModelConf,GeneralConf,ObsConf,NPlot)
    #if config['RunPlotState']:
    #    plot_state_estimation(config)
    #if config['RunPlotForcing']:
    #    plot_forcing_estimation(config)
    #if config['RunPlotParameters']:
    #    plot_parameter_estimation(config)