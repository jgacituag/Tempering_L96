# -*- coding: utf-8 -*-
'''
Nature run to create observations
'''

import sys
import os
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append('../model/')
sys.path.append('../data_assimilation/')
from model import lorenzn as model          # Import the model (fortran routines)
from obsope import common_obs as hoperator  # Import the observation operator (fortran routines)

def initialize_nature_run(conf):
    GeneralConf = conf['GeneralConf']
    ModelConf = conf['ModelConf']
    NatureConf = conf['NatureConf']

    sys.path.append(f"{GeneralConf['FortranRoutinesPath']}/model/")
    sys.path.append(f"{GeneralConf['FortranRoutinesPath']}/data_assimilation/")
    from model import lorenzn as model
    from obsope import common_obs as hoperator

    return GeneralConf, ModelConf, NatureConf

def run_spin_up(ModelConf, NatureConf, NEns, NCoef, Nx, NxSS, X0, XSS0, RF0, CRF0, C0, XPhi, XSigma, CPhi, CSigma):
    print('Doing Spinup')
    start = time.time()
    
    nt = int(NatureConf['SPLength'] / ModelConf['dt'])
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

def run_nature(ModelConf, NatureConf, ObsConf, XSU, XSSSU, RFSU, CRFSU, NEns, NCoef, Nx, NxSS, XPhi, XSigma, CPhi, CSigma):
    print('Doing Nature Run')
    start = time.time()
    
    X0 = XSU[:, :, -1]
    XSS0 = XSSSU[:, :, -1]
    CRF0 = CRFSU[:, :, -1]
    RF0 = RFSU[:, :, -1]

    nt = int(NatureConf['Length'] / ModelConf['dt'])
    ntout = int(nt / ObsConf['Freq']) + 1

    nature_out = model.tinteg_rk4(
        nens=1, nt=nt, ntout=ntout, x0=X0, xss0=XSS0, rf0=RF0,
        phi=XPhi, sigma=XSigma, c0=C0, crf0=CRF0, cphi=CPhi,
        csigma=CSigma, param=ModelConf['TwoScaleParameters'],
        nx=Nx, ncoef=NCoef, dt=ModelConf['dt'], dtss=ModelConf['dtss']
    )

    [XNature, XSSNature, DFNature, RFNature, SSFNature, CRFNature, CNature] = nature_out
    print('Nature run took', time.time() - start, 'seconds.')

    return XNature, XSSNature, DFNature, RFNature, SSFNature, CRFNature, CNature

def generate_observations(ModelConf, ObsConf, XNature, Nx, ntout):
    print('Generating Observations')
    start = time.time()

    NObs = hoperator.get_obs_number(
        ntype=ObsConf['NetworkType'], nx=Nx, nt=ntout,
        space_density=ObsConf['SpaceDensity'],
        time_density=ObsConf['TimeDensity']
    )

    ObsLoc = hoperator.get_obs_location(
        ntype=ObsConf['NetworkType'], nx=Nx, nt=ntout, no=NObs,
        space_density=ObsConf['SpaceDensity'],
        time_density=ObsConf['TimeDensity']
    )

    ObsType = np.ones(np.shape(ObsLoc)[0]) * ObsConf['Type']
    TLoc = np.arange(1, ntout + 1)

    YObs, YObsMask = hoperator.model_to_obs(
        nx=Nx, no=NObs, nt=ntout, nens=1,
        obsloc=ObsLoc, x=XNature, obstype=ObsType,
        obsval=np.zeros(NObs), obserr=np.ones(NObs),
        xloc=ModelConf['XLoc'], tloc=TLoc,
        gross_check_factor=1.0e9, low_dbz_per_thresh=1.1
    )

    ObsLoc[:, 1] = (ObsLoc[:, 1] - 1) * ObsConf['Freq']
    ObsError = np.ones(np.shape(YObs)) * ObsConf['Error']
    ObsBias = np.ones(np.shape(YObs)) * ObsConf['Bias']

    YObs = hoperator.add_obs_error(
        no=NObs, nens=1, obs=YObs, obs_error=ObsError,
        obs_bias=ObsBias, otype=ObsConf['Type']
    )

    print('Observations took', time.time() - start, 'seconds.')
    
    return YObs, ObsLoc, ObsType, ObsError

def save_output(GeneralConf, NatureConf, XNature, XSSNature, DFNature, RFNature, SSFNature, CNature, YObs, NObs, ObsLoc, ObsType, ObsError):
    if NatureConf['RunSave']:
        FNature = DFNature + RFNature + SSFNature
        
        filename = os.path.join(GeneralConf['DataPath'], GeneralConf['NatureFileName'])
        print('Saving the output to ' + filename)
        start = time.time()

        os.makedirs(GeneralConf['DataPath'], exist_ok=True)

        np.savez(
            filename, XNature=XNature, FNature=FNature, CNature=CNature,
            YObs=YObs, NObs=NObs, ObsLoc=ObsLoc, ObsType=ObsType,
            ObsError=ObsError, ModelConf=ModelConf, NatureConf=NatureConf,
            GeneralConf=GeneralConf, XSSNature=XSSNature
        )

        fileout = os.path.join(GeneralConf['DataPath'], 'XNature.csv')
        np.savetxt(fileout, np.transpose(np.squeeze(XNature)), fmt="%6.2f", delimiter=",")

        fileout = os.path.join(GeneralConf['DataPath'], 'XSSNature.csv')
        np.savetxt(fileout, np.transpose(np.squeeze(XSSNature)), fmt="%6.2f", delimiter=",")
        
        print('Saving took', time.time() - start, 'seconds.')

def Run_Nature_Process(conf):
    GeneralConf, ModelConf, NatureConf = initialize_nature_run(conf)

    NCoef = ModelConf['NCoef']
    NEns = NatureConf['NEns']
    Nx = ModelConf['nx']
    NxSS = ModelConf['nxss']
    
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
    
    for ie in range(NEns):
        X0[:, ie] = ModelConf['Coef'][0] / 2 + np.random.normal(size=Nx)
        for ic in range(NCoef):
            C0[:, ie, ic] = ModelConf['Coef'][ic] + FSpaceAmplitude[ic] * np.cos(FSpaceFreq[ic] * 2 * np.pi * np.arange(Nx) / Nx)
    
    XSU, XSSSU, DFSU, RFSU, SSFSU, CRFSU, CSU = run_spin_up(
        ModelConf, NatureConf, NEns, NCoef, Nx, NxSS, X0, XSS0, RF0, CRF0, C0, XPhi, XSigma, CPhi, CSigma
    )
    
    XNature, XSSNature, DFNature, RFNature, SSFNature
