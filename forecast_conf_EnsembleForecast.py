import numpy as np

#=================================================================
# GENERAL SECTION
#=================================================================

GeneralConf=dict()
GeneralConf['ExpName'] ='1000nens_linear'                       # Experiment name
GeneralConf['GeneralPath'] = '/media/jgacitua/storage/Tempering_L96'
GeneralConf['ExpPath'] = f"{GeneralConf['GeneralPath']}/{GeneralConf['ExpName']}"
GeneralConf['FortranRoutinesPath'] = '/media/jgacitua/storage/DABA/Lorenz_96/'
GeneralConf['DataPath'] = f"{GeneralConf['ExpPath']}/DATA"
GeneralConf['FigPath'] = f"{GeneralConf['ExpPath']}/FIGURES"
   
GeneralConf['DAFileName']='DA_' + GeneralConf['ExpName'] + '.npz'
GeneralConf['NatureFileName']='Nature_' + GeneralConf['ExpName'] + '.npz'

GeneralConf['RunSave']=True                                           #Save the output.
GeneralConf['RunPlot']=True                                           #Plot Diagnostics.
GeneralConf['OutFile']='Forecast_' + GeneralConf['ExpName'] + '.npz'  #Output file containing the forecasts.

#File with the initial conditions
GeneralConf['AssimilationFile']=GeneralConf['DataPath'] +'/'+ GeneralConf['DAFileName']
#File with the nature run (for forecast verification)
GeneralConf['NatureFile']      =GeneralConf['DataPath'] +'/'+ GeneralConf['NatureFileName']
#=================================================================
#  FORECAST SECTION :
#=================================================================

ForConf=dict()

ForConf['FreqOut'] = 4                               #Forecast output frequency (in number of time steps)

ForConf['ForecastLength'] = 4 * 50                   #Maximum forecast lead time (in number of time steps)

ForConf['AnalysisSpinUp'] = 400                      #Analysis cycles to skip befor running the first forecast.


