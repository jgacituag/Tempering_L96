#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap

def polar_twin(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    ax2.set_rlabel_position(45)
    #ax2._r_label_position._t = (0 + 180, 0.0)
    #ax2._r_label_position.invalidate()
    #ax2.grid(False, linestyle='-', linewidth=0.5)

    # Ensure that the original axes tick labels are on top of
    # whatever is plotted in the twinned axes. Tick labels will be drawn twice.
    ax2.set_ylim([-20, 60])
    ax2.set_rorigin(-30)
    ax2.set_rticks(np.arange(0, 51, 10))
    plt.setp(ax2.get_yticklabels(), color='darkgreen', fontweight="bold")
    return ax2

def plot_colored_line(ax, theta, r, cmap, norm):
    points = np.array([theta, r]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create the LineCollection for the colored lines
    lc_colored = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
    lc_colored.set_array(r)
    
    # Create the LineCollection for the black edges
    lc_edges = LineCollection(segments, colors='black', linewidth=3.0)

    ax.add_collection(lc_edges)
    ax.add_collection(lc_colored)
    
    return lc_colored

def plot_for_image(data_in, obs_in, idx_obs_in, tobs_in, hradar):
    # Data for plotting
    data = np.append(data_in, data_in[0])
    obs = np.append(obs_in, obs_in[0])
    idx_obs = [int(i - 1) for i in np.append(idx_obs_in, idx_obs_in[0])]
    data_size = np.shape(data)[0]
    theta = np.linspace(0, 2 * np.pi, data_size)
    
    fig, ax = plt.subplots(figsize=(8, 9), subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(top=0.99, bottom=0.15)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels([int(i) for i in np.arange(1, data_size)], fontweight="bold", fontsize=12)
    
    ax.set_title(f'Time {tobs_in:.3f}', fontsize=16)
    ax.set_ylim([-20, 20])
    ax.set_rorigin(-25)
    ax.set_rticks(np.arange(-10, 16, 5))
    ax.grid(True, linestyle='-', linewidth=0.2)
    
    cax = inset_axes(ax, width="100%", height="5%", loc='lower center', borderpad=-5.0)
    cmap = LinearSegmentedColormap.from_list('RdBu_r', plt.cm.RdBu_r(np.linspace(0, 1, 1000)))
    bounds = np.arange(-15, 16, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    line = plot_colored_line(ax, theta, data, cmap, norm)
    line_colorbar = fig.colorbar(mpl.cm.ScalarMappable(cmap = cmap, norm = norm),
                                 cax = cax, orientation = 'horizontal',spacing = 'uniform',
                                 ticks = np.arange(-10, 16, 5))
    line_colorbar.set_label('Nature Value', fontsize=12, color='SteelBlue', fontweight='bold')
    plt.setp(ax.get_yticklabels(), color='SteelBlue', fontweight="bold")
    if hradar:
        ax2 = polar_twin(ax)
        cax2 = inset_axes(ax2, width="100%", height="5%", loc='lower center', borderpad=-10.5)
        cmap_radar = LinearSegmentedColormap.from_list('gist_ncar', plt.cm.gist_ncar(np.linspace(0, 1, 1000)))
        bounds = np.arange(0, 61, 2)
        obs_norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        scatter = ax2.scatter(theta[idx_obs], obs, label='Obs', c=obs, cmap=cmap_radar, linewidth=0.5, edgecolors='black', norm=obs_norm)
        scatter_colorbar = fig.colorbar(mpl.cm.ScalarMappable(cmap = cmap_radar, norm = obs_norm),
                                        cax = cax2, orientation = 'horizontal',spacing = 'uniform',
                                        ticks = np.arange(0, 61, 10))
        scatter_colorbar.set_label('Obs Value', fontsize=12, color='DarkGreen', fontweight='bold')
    else:
        ax.plot(theta[idx_obs], obs, label='Obs', color='SteelBlue', linewidth=2.5, marker='o', markerfacecolor='SteelBlue', markeredgewidth=1.5, markeredgecolor='black')

    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.setp(ax.get_xticklabels(), fontweight="bold")
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
    if hradar:
        plt.pcolor(tmpobs[:, -NPlot:], vmin=0, vmax=60, cmap='gist_ncar')
    else:
        plt.pcolor(tmpobs[:, -NPlot:], vmin=-15, vmax=15, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Observation location')
    plt.title('Observations')
    plt.savefig(FigPath + '/Nature_run_Y.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.plot(tmpobs[0, -NPlot:], 'o', alpha=0.8, markeredgewidth=1.5, markeredgecolor='black')
    plt.plot(XNature[0, 0, -NPlot:])
    plt.xlabel('Time')
    plt.ylabel('X at 1st grid point')
    plt.title('Nature and Observations')
    plt.savefig(FigPath + '/Nature_and_Obs_Time_Serie.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.plot(xobs[:, -1], tmpobs[:, -1], 'o', alpha=0.8, markeredgewidth=1.5, markeredgecolor='black')
    plt.plot(np.arange(1, Nx + 1), XNature[:, 0, -1], '-o', alpha=0.8, markeredgewidth=1.5, markeredgecolor='black')
    plt.xlabel('Location')
    plt.ylabel('X at last time')
    plt.title('Nature and Observations')
    plt.savefig(FigPath + '/Nature_and_Obs_At_Last_Time.png', facecolor='w', format='png')
    plt.show(block=False)
    plt.close()
    
    # Create gif with nature and observations
    images = []
    for i in range(NPlot):
        img = plot_for_image(XNature[:, 0, i], tmpobs[:, i], xobs[:, i], ModelConf['dt'] * np.unique(tobs)[i], hradar)
        images.append(img)
        
    imageio.mimsave(FigPath + '/Nature_run_X.gif', images, fps=6)
