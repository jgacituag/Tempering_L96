# -*- coding: utf-8 -*-
'''
Template to create configurations
'''

import numpy as np

#=============================================================================#
#                            GENERAL CONFIGURATION                            #
#=============================================================================#

GeneralConf=dict()

GeneralConf['ExpName'] ='fancy_and_overexplicative_name'                       # Experiment name
GeneralConf['ExpPath'] = f"path/to/expdir/{GeneralConf['ExpName']}"            # path to the experiment
GeneralConf['NewExperiment'] = True
GeneralConf['NewExperiment'] = True
GeneralConf['NewExperiment'] = True
GeneralConf['NewExperiment'] = True


#=============================================================================#
#                                MODEL SECTION                                #
#=============================================================================#
#General model section

ModelConf=dict()

#General model section

ModelConf['nx'] =  40                                   #Number of large-scale state variables
ModelConf['dt']  =0.0125                                #Time step for large-scale variables (do not change)

#Forcing section

ModelConf['Coef']=np.array([8.0])
ModelConf['NCoef']=np.size(ModelConf['Coef'])

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
#  DATA ASSIMILATION SECTION :
#=============================================================================#

DAConf=dict()

DAConf['ExpLength'] = 1000                           #None use the full nature run experiment. Else use this length.

DAConf['NEns'] = 30                                  #Number of ensemble members

DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.

DAConf['Freq'] = 4                                   #Assimilation frequency (in number of time steps)
DAConf['TSFreq'] = 4                                 #Intra window ensemble output frequency (for 4D Data assimilation)

DAConf['InfCoefs']=np.array([1.04,0.0,0.0,0.0,0.0])  #Mult inf, RTPS, RTPP, EPES, Additive inflation

DAConf['LocScalesLETKF']=np.array([4.0,-1.0])        #Localization scale is space and time (negative means no localization)
DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)

#Initial state ensemble.
DAConf['InitialXSigma']=0.5                          #Initial ensemble spread for state variables.

DAConf['UpdateSmoothCoef']=0.0                       #Data assimilation update smooth (for parameter estimation only)

#Parameter estimation/perturbation 

DAConf['InitialPSigma']=np.array([0,0,0])            #Initial ensemble spread for the parameters. (0 means no parameter estimation)

DAConf['GrossCheckFactor'] = 20.0                    #Gross check error threshold (observations associated with innovations greather than GrossCheckFactor * R**0.5 will be rejected).
DAConf['LowDbzPerThresh']  = 1.01                    #If the percentaje of ensemble members with reflectivity values == to the lower limit, then the observation is not assimilated [reflectivity obs only]



