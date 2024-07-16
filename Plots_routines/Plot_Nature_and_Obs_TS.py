#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def polar_twin(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    ax2._r_label_position._t = (22.5 + 180, 0.0)
    ax2._r_label_position.invalidate()

    # Bit of a hack to ensure that the original axes tick labels are on top of
    # whatever is plotted in the twinned axes. Tick labels will be drawn twice.
    for label in ax.get_yticklabels():
        ax.figure.texts.append(label)
    ax2.set_ylim([0, 60])
    plt.setp(ax2.get_yticklabels(), color='darkgreen')
    return ax2

def plot_for_image(data_in, obs_in, idx_obs_in, tobs_in, hradar):
    # Data for plotting
    data = np.append(data_in, data_in[0])
    obs = np.append(obs_in, obs_in[0])
    idx_obs = [int(i - 1) for i in np.append(idx_obs_in, idx_obs_in[0])]
    data_size = np.shape(data)[0]
    theta = np.linspace(0, 2 * np.pi, data_size)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.legend(loc="upper left", bbox_to_anchor=(0.9, 1.05))
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels([int(i) for i in np.arange(1, data_size)])
    
    ax.set_title(f'Time {tobs_in:.3f}', fontsize=16)
    #ax.set_rmax(20)
    #ax.set_rmin(-20)
    ax.set_ylim([-20, 20])
    ax.set_rorigin(-25)
    ax.set_rticks(np.arange(-15, 16, 5))
    ax.grid(True)

    ax.plot(theta, data, label='Nature', color='IndianRed', linewidth=2.5)
    
    if hradar:
        ax2 = polar_twin(ax)
        scatter = ax2.scatter(theta[idx_obs], obs, label='Obs', c=obs, cmap='gist_ncar', linewidth=2.5)
        
        # Create an inset axis for the colorbar
        cax = inset_axes(ax2, width="5%", height="50%", loc='upper right', borderpad=1)
        fig.colorbar(scatter, cax=cax, orientation='vertical')
    else:
        ax.plot(theta[idx_obs], obs, label='Obs', color='SteelBlue', linewidth=2.5)

    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image

def plot_nature_and_obs(NatureDataPath, ModelConf, GeneralConf, ObsConf, NPlot):
    if not isinstance(NPlot, int):
        NPlot = 1000  # Plot the last NPlot times.
    
    hradar = ObsConf['Type'] == 3
    
    Data = np.load(NatureDataPath, allow_pickle=True)
    XNature = Data['XNature']
    YObs = Data['YObs']
    ObsLoc = Data['ObsLoc']
    Nx = ModelConf['nx']
    FigPath = GeneralConf['FiguresPath']
    os.makedirs(FigPath, exist_ok=True)
    print('Plotting the output')

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
    plt.savefig(FigPath + '/Nature_run_X.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.pcolor(tmpobs[:, -NPlot:], vmin=-15, vmax=15, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Observation location')
    plt.title('Observations')
    plt.savefig(FigPath + '/Nature_run_Y.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.plot(tmpobs[0, -NPlot:], 'o', alpha=0.8)
    plt.plot(XNature[0, 0, -NPlot:])
    plt.xlabel('Time')
    plt.ylabel('X at 1st grid point')
    plt.title('Nature and Observations')
    plt.savefig(FigPath + '/Nature_and_Obs_Time_Serie.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.plot(xobs[:, -1], tmpobs[:, -1], 'o', alpha=0.8)
    plt.plot(np.arange(1, Nx + 1), XNature[:, 0, -1], '-o', alpha=0.8)
    plt.xlabel('Location')
    plt.ylabel('X at last time')
    plt.title('Nature and Observations')
    plt.savefig(FigPath + '/Nature_and_Obs_At_Last_Time.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()
    
    # Create gif with nature and observations
    images = [plot_for_image(XNature[:, 0, i], tmpobs[:, i], xobs[:, i], ModelConf['dt'] * np.unique(tobs)[i], hradar) for i in range(NPlot)]
    imageio.mimsave(FigPath + '/Nature_run_X.gif', images, fps=10)
