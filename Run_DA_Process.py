# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import numpy as np
import experiment_config as expconf
sys.path.append(f"{expconf.GeneralConf['FortranRoutinesPath']}/model/")
sys.pa.append(f"{expconf.GeneralConf['FortranRoutinesPath']}/data_assimilation/")
from model import lorenzn as model          #Import the model (fortran routines)
from obsope import common_obs as hoperator  #Import the observation operator (fortran routines)
from da import common_da_tools as das       #Import the data assimilation routines (fortran routines)


def get_temp_steps( NTemp , Alpha ) :
    
   #NTemp is the number of tempering steps to be performed.
   #Alpha is a slope coefficient. Larger alpha means only a small part of the information
   #will be assimilated in the first step (and the largest part will be assimilated in the last step).

   dt=1.0/float(NTemp+1)
   steps = np.exp( 1.0 * Alpha / np.arange( dt , 1.0-dt/100.0 , dt ) )
   steps = steps / np.sum(steps)

   return steps 

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

def verification_diagnostics(XA,XF,XNature,DALength,SpinUp=200):
    XASpread = np.std(XA,axis=1)
    XFSpread = np.std(XF,axis=1)

    XAMean = np.mean(XA,axis=1)
    XFMean = np.mean(XF,axis=1)

    XASRmse = np.sqrt(np.mean(np.power(XAMean[:, SpinUp:DALength] - XNature[:,0,SpinUp:DALength], 2), axis=1 ) )
    XFSRmse = np.sqrt(np.mean(np.power(XFMean[:, SpinUp:DALength] - XNature[:,0,SpinUp:DALength], 2), axis=1 ) )

    XATRmse = np.sqrt(np.mean(np.power(XAMean - XNature[:,0,0:DALength], 2), axis = 0))
    XFTRmse = np.sqrt(np.mean(np.power(XFMean - XNature[:,0,0:DALength], 2), axis = 0))

    XASBias = np.mean(XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength], axis=1)
    XFSBias = np.mean(XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength], axis=1)

    XATBias = np.mean(XAMean-XNature[:,0,0:DALength], axis=0)
    XFTBias = np.mean(XFMean-XNature[:,0,0:DALength], axis=0)

    print(' Analysis RMSE ',np.mean(XASRmse),' Analysis SPREAD ',np.mean(XASpread))

    return XAMean, XFMean, XASpread, XFSpread, XASRmse, XFSRmse, XATRmse, XFTRmse, XASBias, XFSBias, XATBias, XFTBias

def run_da(GeneralConf,ModelConf,DAConf,ObsConf):

    #=================================================================
    #  LOAD OBSERVATIONS AND NATURE RUN CONFIGURATION
    #=================================================================
    ObsFile = ObsConf['obsfile']
    logging.info(f"Reading observations from file {ObsFile}")
        
    InputData=np.load(ObsFile,allow_pickle=True)
        
    DAConf['Freq']=ObsConf['Freq']
    DAConf['TSFreq']=ObsConf['Freq']

    YObs = InputData['YObs']          #Obs value
    ObsLoc = InputData['ObsLoc']      #Obs location (space , time)
    ObsType = InputData['ObsType']    #Obs type ( x or x^2)
    ObsError = InputData['ObsError']  #Obs error 

    ModelConf['dt'] = InputData['ModelConf'][()]['dt']
        
    #Store the true state evolution for verfication
    XNature = InputData['XNature']    #State variables
    CNature = InputData['CNature']    #Parameters
    FNature = InputData['FNature']    #Large scale forcing.

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
    F=np.zeros([Nx,NEns,DALength])                          #Total forcing on large scale variables.

    XSigma = ModelConf['XSigma'] if ModelConf['EnableSRF'] else 0.0
    XPhi = ModelConf['XPhi'] if ModelConf['EnableSRF'] else 1.0
    CSigma = ModelConf['CSigma'] if ModelConf['EnablePRF'] else np.zeros(NCoef)
    CPhi = ModelConf['CPhi'] if ModelConf['EnablePRF'] else 1.0
    FSpaceAmplitude = ModelConf['FSpaceAmplitude'] if ModelConf['FSpaceDependent'] else np.zeros(NCoef)
    FSpaceFreq = ModelConf['FSpaceFreq']
    
    CRF = np.zeros([NEns, NCoef])
    RF = np.zeros([Nx, NEns])
    XSS = np.zeros((NxSS, NEns))
    SFF=np.zeros((Nx,NEns))
    C0 = np.zeros((Nx, NEns, NCoef))
    
    #Generate a random initial conditions and initialize deterministic parameters
    for ie in range(0,NEns):
        RandInd1=(np.round(np.random.rand(1)*DALength)).astype(int)
        RandInd2=(np.round(np.random.rand(1)*DALength)).astype(int)
        
        #Replace the random perturbation for a mos inteligent perturbation
        XA[:,ie,0]=ModelConf['Coef'][0]/2 + np.squeeze( DAConf['InitialXSigma'] * ( XNature[:,0,RandInd1] - XNature[:,0,RandInd2] ) )

        for ic in range(0,NCoef) : 
            PA[:,ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=1 )
                    


    logging.info(f'Running Data Assimilation')
    for it in range(1, DALength):
        if np.mod(it,100) == 0:
            logging.info(f'Data assimilation cycle # {it}')
            print('Data assimilation cycle # ',str(it) )
        #=================================================================#
        #                        ENSEMBLE FORECAST                        #
        #=================================================================#
        #Run the ensemble forecast
        #print('Runing the ensemble')

        ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.
        NT = DAConf['Freq']
        DT = ModelConf['dt']
        X0 = XA[:,:,it-1]
        C0 = PA[:,:,:,it-1]
        params = ModelConf['TwoScaleParameters']
        DTSS = ModelConf['dtss']
        ensout=model.tinteg_rk4(nens=NEns, nt=NT,  ntout=ntout,
                                x0=X0, xss0=XSS, rf0=RF, phi=XPhi, sigma=XSigma,
                                c0=C0, crf0=CRF, cphi=CPhi, csigma=CSigma, param=params,
                                nx=Nx,  nxss=NxSS, ncoef=NCoef, dt=DT, dtss=DTSS)
        [ XFtmp , XSStmp , DFtmp , RFtmp , SSFtmp , CRFtmp, CFtmp ] = ensout
        PF[:,:,:,it] = CFtmp[:,:,:,-1]  #Store the parameter at the end of the window
        XF[:,:,it] = XFtmp[:,:,-1]      #Store the state variables ensemble at the end of the window
            
        F[:,:,it] = DFtmp[:,:,-1]+RFtmp[:,:,-1]+SSFtmp[:,:,-1]  #Store the total forcing
            
        XSS = XSStmp[:,:,-1]
        CRF = CRFtmp[:,:,-1]
        RF = RFtmp[:,:,-1]
            
        #=================================================================#
        #           GET THE OBSERVATIONS WITHIN THE TIME WINDOW           #
        #=================================================================#

        da_window_start  = (it -1) * DAConf['Freq']
        da_window_end    = da_window_start + DAConf['Freq']
        da_analysis_time = da_window_end

        #Screen the observations and get only the one within the da window
        window_mask=np.logical_and( ObsLoc[:,1] > da_window_start , ObsLoc[:,1] <= da_window_end )

        ObsLocW=ObsLoc[window_mask,:]                                     #Observation location within the DA window.
        ObsTypeW=ObsType[window_mask]                                     #Observation type within the DA window
        YObsW=YObs[window_mask]                                           #Observations within the DA window
        NObsW=YObsW.size                                                  #Number of observations within the DA window
        ObsErrorW=ObsError[window_mask]                                   #Observation error within the DA window  

        #=================================================================#
        #                       HYBRID-TEMPERED DA                        #
        #=================================================================#

        stateens = np.copy(XF[:,:,it])

        #=================================================================#
        #                      OBSERVATION OPERATOR                       #
        #=================================================================#      
        #Apply h operator and transform from model space to observation space.
        #This opearation is performed only at the end of the window.

        if NObsW > 0:
            XLOC=ModelConf['XLoc']
            TLoc= da_window_end #We are assuming that all observations are valid at the end of the assimilaation window.
            [YF, YFqc] = hoperator.model_to_obs(nx = Nx, no = NObsW, nt = 1, nens = NEns,
                            obsloc = ObsLocW, x = stateens, obstype=ObsTypeW, obserr=ObsErrorW, obsval=YObsW,
                            xloc = XLOC, tloc = TLoc, gross_check_factor = DAConf['GrossCheckFactor'],
                            low_dbz_per_thresh = DAConf['LowDbzPerThresh'])
            YFmask = np.ones( YFqc.shape ).astype(bool)
            YFmask[YFqc != 1] = False

            ObsLocW = ObsLocW[YFmask, :]
            ObsTypeW = ObsTypeW[YFmask]
            YObsW = YObsW[YFmask, :]
            NObsW = YObsW.size
            ObsErrorW = ObsErrorW[YFmask, :]
            YF = YF[YFmask, :]
   
            #=================================================================
            #  LETKF STEP  : 
            #=================================================================

            stateens = das.da_letkf(nx = Nx ,nt = 1 , no = NObsW, nens = NEns, xloc = XLOC,
                                    tloc = da_window_end, nvar = 1, xfens = stateens,
                                    obs = YObsW ,obsloc = ObsLocW, ofens = YF,
                                    rdiag = ObsErrorW , loc_scale = DAConf['LocScalesLETKF'], inf_coefs= DAConf['InfCoefs'],
                                    update_smooth_coef = 0.0 , temp_factor = np.ones(Nx) )[:,:,0,0]
   
            XA[:,:,it] = np.copy(stateens)
            PA[:,:,:,it]=PA[:,:,:,0]

    verout = verification_diagnostics(XA,XF,XNature,DALength)
    XAMean, XFMean, XASpread, XFSpread, XASRmse, XFSRmse, XATRmse, XFTRmse, XASBias, XFSBias, XATBias, XFTBias = verout
    return XA, PA, F, XF, XAMean, XFMean, XASpread, XFSpread, XASRmse, XFSRmse, XATRmse, XFTRmse, XASBias, XFSBias, XATBias, XFTBias


def save_output(DAout, ModelConf, DAConf, ObsConf,GeneralConf):
    XA, PA, F, XF, XAMean, XFMean, XASpread, XFSpread, XASRmse, XFSRmse, XATRmse, XFTRmse, XASBias, XFSBias, XATBias, XFTBias = DAout
    filename = os.path.join(GeneralConf['DataPath'], DAConf['DAFileName'])
    logging.info(f'Saving the output to {filename}')
    start = time.time()
    os.makedirs(GeneralConf['DataPath'], exist_ok=True)

    #Save Nature run output
    np.savez(filename,
             XA = XA, PA = PA, F = F, XF = XF, 
             XAMean = XAMean, XFMean = XFMean,
             XASpread = XASpread, XFSpread = XFSpread,
             XASRmse = XASRmse, XFSRmse = XFSRmse,
             XATRmse = XATRmse, XFTRmse = XFTRmse ,
             XASBias = XASBias, XFSBias = XFSBias,
             XATBias = XATBias, XFTBias = XFTBias,
             ModelConf = ModelConf   , DAConf = DAConf, GeneralConf = GeneralConf, ObsConf = ObsConf)
    logging.info(f'Saving took {time.time() - start} seconds.')

def run_plot_process(conf):
    GeneralConf = conf.GeneralConf
    DAConf = conf.DAConf
    ModelConf = conf.ModelConf
    ObsConf = conf.ObsConf

    DAout = run_da(GeneralConf, ModelConf, DAConf, ObsConf)

    if DAConf['RunSave']:
        save_output(DAout, ModelConf, DAConf, ObsConf,GeneralConf)