# -*- coding: utf-8 -*-

import sys
import os
import time
import numpy as np
from scipy import stats
import experiment_config as expconf
sys.path.append(f"{expconf.GeneralConf['FortranRoutinesPath']}/model/")
sys.path.append(f"{expconf.GeneralConf['FortranRoutinesPath']}/data_assimilation/")
from model  import lorenzn          as model          #Import the model (fortran routines)
from obsope import common_obs       as hoperator      #Import the observation operator (fortran routines)
from da     import common_da_tools  as das            #Import the data assimilation routines (fortran routines)


def inflation( ensemble_post , ensemble_prior , nature , inf_coefs )  :

   #This function consideres inflation approaches that are applied after the analysis. In particular when these approaches
   #are used in combination with tempering.
   DALength = nature.shape[2] - 1
   NEns = ensemble_post.shape[1]

   if inf_coefs[5] > 0.0 :
     #=================================================================
     #  RTPS  : Relaxation to prior spread (compatible with tempering iterations) 
     #=================================================================
     prior_spread = np.std( ensemble_prior , axis=1 )
     post_spread  = np.std( ensemble_post  , axis=1 )
     PostMean = np.mean( ensemble_post , axis=1 )
     EnsPert = ensemble_post - np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 )
     inf_factor = ( 1.0 - inf_coefs[5] ) + ( prior_spread / post_spread ) * inf_coefs[5]
     EnsPert = EnsPert * np.repeat( inf_factor[:,np.newaxis] , NEns , axis=1 )
     ensemble_post = EnsPert + np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 )

   if inf_coefs[6] > 0.0 :
     #=================================================================
     #  RTPP  : Relaxation to prior perturbations (compatible with tempering iterations) 
     #=================================================================
     PostMean = np.mean( ensemble_post , axis=1 )
     PriorMean= np.mean( ensemble_prior, axis=1 )
     PostPert = ensemble_post - np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 )
     PriorPert= ensemble_prior- np.repeat( PriorMean[:,np.newaxis] , NEns , axis=1 )
     PostPert = (1.0 - inf_coefs[6] ) * PostPert + inf_coefs[6] * PriorPert 
     ensemble_post = PostPert + np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 ) 

   if inf_coefs[4] > 0.0 :
     #=================================================================
     #  ADD ADDITIVE ENSEMBLE PERTURBATIONS  : 
     #=================================================================
     #Additive perturbations will be generated as scaled random
     #differences of nature run states.
     #Get random index to generate additive perturbations
     RandInd1=(np.round(np.random.rand(NEns)*DALength)).astype(int)
     RandInd2=(np.round(np.random.rand(NEns)*DALength)).astype(int)
     AddInfPert = np.squeeze( nature[:,0,RandInd1] - nature[:,0,RandInd2] ) * inf_coefs[4]
     #Shift perturbations to obtain zero-mean perturbations and add it to the ensemble.
     ensemble_post = ensemble_post + AddInfPert - np.repeat( np.mean(AddInfPert,1)[:,np.newaxis] , NEns , axis=1 )

   return ensemble_post

def run_spin_up(ModelConf, DAConf, NEns, NCoef, Nx, NxSS, X0, XSS0, RF0, CRF0, C0, XPhi, XSigma, CPhi, CSigma):
    print('Doing Spinup')
    start = time.time()
    
    nt = int(DAConf['SPLength'] / ModelConf['dt'])
    ntout = 2

    spin_up_out = model.tinteg_rk4(
        nens=1, nt=nt, ntout=ntout, x0=X0, xss0=XSS0, rf0=RF0,
        phi=XPhi, sigma=XSigma, c0=C0, crf0=CRF0, cphi=CPhi,
        csigma=CSigma, param=ModelConf['TwoScaleParameters'],
        nx=Nx, ncoef=NCoef, dt=ModelConf['dt'], dtss=ModelConf['dtss']
    )
    
    [XSU, XSSSU, DFSU, RFSU, SSFSU, CRFSU, CSU] = spin_up_out   
    print('Spinup took', time.time() - start, 'seconds.')
    
    return XSU, XSSSU, DFSU, RFSU, SSFSU, CRFSU, CSU

def run_da(ModelConf, DAConf, ObsConf, XSU, XSSSU, RFSU, CRFSU, NEns, NCoef, Nx, NxSS, XPhi, XSigma, CPhi, CSigma):
    print('Running Data Assimilation')
    start = time.time()
    
    X0 = XSU[:, :, -1]
    XSS0 = XSSSU[:, :, -1]
    CRF0 = CRFSU[:, :, -1]
    RF0 = RFSU[:, :, -1]

    nt = int(DAConf['Length'] / ModelConf['dt'])
    ntout = int(nt / ObsConf['Freq']) + 1

    da_out = model.tinteg_rk4(
        nens=1, nt=nt, ntout=ntout, x0=X0, xss0=XSS0, rf0=RF0,
        phi=XPhi, sigma=XSigma, c0=C0, crf0=CRF0, cphi=CPhi,
        csigma=CSigma, param=ModelConf['TwoScaleParameters'],
        nx=Nx, ncoef=NCoef, dt=ModelConf['dt'], dtss=ModelConf['dtss']
    )

    [XDA, XSSDA, DFDA, RFDA, SSFDA, CRFDA, CDA] = da_out
    print('Data assimilation took', time.time() - start, 'seconds.')

    return XDA, XSSDA, DFDA, RFDA, SSFDA, CRFDA, CDA


def save_output(GeneralConf, DAConf, XDA, XSSDA, DFDA, RFDA, SSFDA, CDA, YObs, NObs, ObsLoc, ObsType, ObsError):
    if DAConf['RunSave']:
        FDA = DFDA + RFDA + SSFDA
        
        filename = os.path.join(GeneralConf['DataPath'], GeneralConf['DAFileName'])
        print('Saving the output to ' + filename)
        start = time.time()

        os.makedirs(GeneralConf['DataPath'], exist_ok=True)

        np.savez(
            filename, XDA=XDA, FDA=FDA, CDA=CDA,
            YObs=YObs, NObs=NObs, ObsLoc=ObsLoc, ObsType=ObsType,
            ObsError=ObsError, ModelConf=ModelConf, DAConf=DAConf,
            GeneralConf=GeneralConf, XSSDA=XSSDA
        )

        fileout = os.path.join(GeneralConf['DataPath'], 'XDA.csv')
        np.savetxt(fileout, np.transpose(np.squeeze(XDA)), fmt="%6.2f", delimiter=",")

        fileout = os.path.join(GeneralConf['DataPath'], 'XSSDA.csv')
        np.savetxt(fileout, np.transpose(np.squeeze(XSSDA)), fmt="%6.2f", delimiter=",")
        
        print('Saving took', time.time() - start, 'seconds.')

def run_da_process(conf):
    GeneralConf = conf.GeneralConf
    DAConf = conf.DAConf
    ModelConf = conf.ModelConf

    #=================================================================
    #  LOAD OBSERVATIONS AND NATURE RUN CONFIGURATION
    #=================================================================
        
    print('Reading observations from file ',GeneralConf['ObsFile'])
        
    InputData=np.load(GeneralConf['ObsFile'],allow_pickle=True)
        
    ObsConf=InputData['ObsConf'][()]
    DAConf['Freq']=ObsConf['Freq']
    DAConf['TSFreq']=ObsConf['Freq']

    YObs    =  InputData['YObs']         #Obs value
    ObsLoc  =  InputData['ObsLoc']       #Obs location (space , time)
    ObsType =  InputData['ObsType']      #Obs type ( x or x^2)
    ObsError=  InputData['ObsError']     #Obs error 

    ModelConf['dt'] = InputData['ModelConf'][()]['dt']
        
    #Store the true state evolution for verfication 
    XNature = InputData['XNature']   #State variables
    CNature = InputData['CNature']   #Parameters
    FNature = InputData['FNature']   #Large scale forcing.

    if DAConf['ExpLength'] == None :
        DALength = int( max( ObsLoc[:,1] ) / DAConf['Freq'] )
    else:
        DALength = DAConf['ExpLength']
        XNature = XNature[:,:,0:DALength+1]
        CNature = CNature[:,:,:,0:DALength+1] 
        FNature = FNature[:,:,0:DALength+1]
   
    NCoef = ModelConf['NCoef']
    NEns = DAConf['NEns']
    Nx = ModelConf['nx']
    NxSS = ModelConf['nxss']
    
    XA=np.zeros([Nx,NEns,DALength])                         #Analisis ensemble
    XF=np.zeros([Nx,NEns,DALength])                         #Forecast ensemble
    PA=np.zeros([Nx,NEns,NCoef,DALength])                   #Analized parameters
    PF=np.zeros([Nx,NEns,NCoef,DALength])                   #Forecasted parameters
    NAssimObs=np.zeros(DALength)
        
    F=np.zeros([Nx,NEns,DALength])                          #Total forcing on large scale variables.
    

    XSigma = ModelConf['XSigma'] if ModelConf['EnableSRF'] else 0.0
    XPhi = ModelConf['XPhi'] if ModelConf['EnableSRF'] else 1.0
    CSigma = ModelConf['CSigma'] if ModelConf['EnablePRF'] else np.zeros(NCoef)
    CPhi = ModelConf['CPhi'] if ModelConf['EnablePRF'] else 1.0
    FSpaceAmplitude = ModelConf['FSpaceAmplitude'] if ModelConf['FSpaceDependent'] else np.zeros(NCoef)
    FSpaceFreq = ModelConf['FSpaceFreq']
    
    CRF0 = np.zeros([NEns, NCoef])
    RF0 = np.zeros([Nx, NEns])
    X0 = np.zeros((Nx, NEns))
    XSS0 = np.zeros((NxSS, NEns))
    C0 = np.zeros((Nx, NEns, NCoef))
    
    #Generate a random initial conditions and initialize deterministic parameters
    for ie in range(0,NEns):
        RandInd1=(np.round(np.random.rand(1)*DALength)).astype(int)
        RandInd2=(np.round(np.random.rand(1)*DALength)).astype(int)
        
        #Replace the random perturbation for a mos inteligent perturbation
        XA[:,ie,0]=ModelConf['Coef'][0]/2 + np.squeeze( DAConf['InitialXSigma'] * ( XNature[:,0,RandInd1] - XNature[:,0,RandInd2] ) )

        for ic in range(0,NCoef) : 
            PA[:,ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=1 )
                    
    XSU, XSSSU, DFSU, RFSU, SSFSU, CRFSU, CSU = run_spin_up(
        ModelConf, DAConf, NEns, NCoef, Nx, NxSS, X0, XSS0, RF0, CRF0, C0, XPhi, XSigma, CPhi, CSigma
    )
    
    XDA, XSSDA, DFDA, RFDA, SSFDA, CRFDA, CDA = run_da(ModelConf, DAConf, ObsConf, XSU, XSSSU, RFS)
