# -*- coding: utf-8 -*-
import numpy as np
import os

#=============================================================================#
#                            GENERAL CONFIGURATION                            #
#=============================================================================#

GeneralConf=dict()

GeneralConf['ExpName'] ='test_experiment'                       # Experiment name
GeneralConf['GeneralPath'] = '/media/jgacitua/storage/Tempering_L96'
GeneralConf['ExpPath'] = f"{GeneralConf['GeneralPath']}/{GeneralConf['ExpName']}"            # path to the experiment
#GeneralConf['FortranRoutinesPath'] = '/media/jgacitua/storage/Tempering_L96/Fortran_routines'               # path to the compiled fortran routines
GeneralConf['FortranRoutinesPath'] = '/media/jgacitua/storage/DABA/Lorenz_96/'
GeneralConf['DataPath'] = f"{GeneralConf['ExpPath']}/DATA"
GeneralConf['FiguresPath'] = f"{GeneralConf['ExpPath']}/FIGURES"
GeneralConf['NewExperiment'] = True
GeneralConf['RunNature'] = True
GeneralConf['RunDA'] = False
GeneralConf['RunPlots'] = True
GeneralConf['MultipleConfigurations'] = False
GeneralConf['MultivalueVariables'] = [('ObsConf', 'SpaceDensity'), 
                                      ('ObsConf', 'Error')]

#=============================================================================#
#                                MODEL SECTION                                #
#=============================================================================#

ModelConf=dict()

#General model section
ModelConf['nx'] = 40                                    #Number of large-scale state variables
ModelConf['dt'] = 0.0125                                #Time step for large-scale variables (do not change)
#Forcing section
ModelConf['Coef']=np.array([8])                         #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... ) 
ModelConf['NCoef']=np.size(ModelConf['Coef'])           #Get the total number of coefs.

#Space dependent parameter
ModelConf['FSpaceDependent']=False                      #If the forcing parameters will depend on the location.
ModelConf['FSpaceAmplitude']=np.array([1])              #Amplitude of space variantions (for each coefficient)
ModelConf['FSpaceFreq']     =np.array([1])              #Use integers >= 1

#Parameter random walk          
ModelConf['EnablePRF']=False                            #Activate Parameter random walk
ModelConf['CSigma']=np.array([0])                       #Parameter random walk sigma
ModelConf['CPhi'  ]=1.0                                 #Parameter random walk phi

#State random forcing parameters
ModelConf['EnableSRF']=False                            #Activate State random forcing.
ModelConf['XSigma']=0.0                                 #Amplitude of the random walk
ModelConf['XPhi'  ]=1.0                                 #Time autocorrelation parameter
ModelConf['XLoc'  ]=np.arange(1,ModelConf['nx']+1)      #Location of model grid points (1-nx)

#Two scale model parameters
ModelConf['TwoScaleParameters']=np.array([10,10,0])     #Small scale and coupling parameters C , B and Hint
                                                        #Set Hint /= 0 to enable two scale model                                              
ModelConf['nxss']= ModelConf['nx'] * 8                  #Number of small scale variables
ModelConf['dtss']= ModelConf['dt'] / 5                  #Time step increment for the small scale variables

#=============================================================================#
#                                NATURE SECTION                               #
#=============================================================================#

NatureConf= dict()
NatureConf['NatureFileName']='Nature_' + GeneralConf['ExpName'] + '.npz'
NatureConf['NEns']=1                                   # Number of ensemble members for the nature run. (usually 1)
NatureConf['RunSave']=True                             # Save nature run
NatureConf['RunPlot']=True                             # Plot nature run
NatureConf['SPLength']=40                              # Spin up length in model time units (1 model time unit app. equivalent to 5 day time in the atmosphere)
NatureConf['Length']=1000                              # Nature run length in model time units (1 model time unit app. equivalent to 5 day time in the atmosphere)

#Nature plots if NatureConf['RunPlot']=True
NatureConf['NPlot']=500
NatureConf['RunPlotNatureHoperator']=True              # Plot the nature run and the observation operator
NatureConf['RunPlotNatureObsGIF']=True                 # Plot the nature run and the observations in a gif
#=============================================================================#
#                            OBSERVATIONS SECTION                             #
#=============================================================================#

ObsConf= dict()

ObsConf['Freq']=4                                      # Observation frequency in number of time steps (will also control nature run output frequency)
ObsConf['obsfile'] = os.path.join(GeneralConf['DataPath'], NatureConf['NatureFileName'])
#Observation location
ObsConf['NetworkType']='regular'                       # Observation network type: REGULAR, RANDOM, FROMFILE
ObsConf['SpaceDensity']= 1 #np.array([0.5, 1])        # Observation density in space. Usually from [0-1] but can be greater than 1.
ObsConf['TimeDensity']=1                               # Observation density in time. Usually from [0-1] but can be greater than 1.
                                                       # Do not use ObsTimeDensity to change observation frequency for REGULAR obs, use ObsFreq instead.
#Set the diagonal of R
ObsConf['Error']=1 #np.array([0.9, 1, 1.1])               # Constant observation error.
ObsConf['Bias']=0.0                                    # Constant Systematic observation error.
ObsConf['Type']=1                                      # Observation type (1 observe x, 2 observe x**2)

#=================================================================
#  DATA ASSIMILATION SECTION :
#=================================================================

DAConf=dict()
DAConf['RunSave']=True                                  # Save DA run
DAConf['RunPlot']=True                                  # Plot DA run
DAConf['DAFileName']='DA_' + GeneralConf['ExpName'] + '.npz'
DAConf['ExpLength'] = 1000                              # None use the full nature run experiment. Else use this length.
DAConf['NEns'] = 30                                     # Number of ensemble members
DAConf['Twin'] = True                                   # When True, model configuration will be replaced by the model configuration in the nature run.
DAConf['Freq'] = 4                                      # Assimilation frequency (in number of time steps)
DAConf['TSFreq'] = 4                                    # Intra window ensemble output frequency (for 4D Data assimilation)
DAConf['InfCoefs']=np.array([1.04,0.0,0.0,0.0,0.0])     # Mult inf, RTPS, RTPP, EPES, Additive inflation
DAConf['LocScalesLETKF']=np.array([4.0,-1.0])           # Localization scale is space and time (negative means no localization)
DAConf['LocScalesLETPF']=np.array([3.0,-1.0])           # Localization scale is space and time (negative means no localization)

#Initial state ensemble.
DAConf['InitialXSigma']=0.5                             # Initial ensemble spread for state variables.
DAConf['UpdateSmoothCoef']=0.0                          # Data assimilation update smooth (for parameter estimation only)

#Parameter estimation/perturbation 
DAConf['InitialPSigma']=np.array([0,0,0])               # Initial ensemble spread for the parameters. (0 means no parameter estimation)
DAConf['GrossCheckFactor'] = 20.0                       # Gross check error threshold (observations associated with innovations greater than GrossCheckFactor * R**0.5 will be rejected).
DAConf['LowDbzPerThresh']  = 1.01                       # If the percentage of ensemble members with reflectivity values == to the lower limit, then the observation is not assimilated [reflectivity obs only]
