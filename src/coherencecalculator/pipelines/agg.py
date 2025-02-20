from coherencecalculator.pipelines.timeseries import timeseries
import coherencecalculator.tools.utils as utils
import numpy as np
import pandas as pd
from coherencecalculator.tools.vecloader import VecLoader

def agg(vecLoader:VecLoader, inputTimeseries=None, inputText=None, inputDir=None, inputCsv=None, inputPickle=None, inputDf=None, fileCol=None, textCol=None, vecType=None, saveDir=None, aggType=np.min) -> pd.DataFrame:
    if inputTimeseries is None:
        cosineDf = timeseries(vecLoader, inputText=inputText, inputDir=inputDir, inputCsv=inputCsv, inputPickle=inputPickle, inputDf=inputDf, fileCol=fileCol, textCol=textCol, vecType=vecType, saveDir=None)
    else:
        cosineDf = inputTimeseries.copy()
    
    allTsCols = [col for col in cosineDf.columns if col not in ['file', 'text', 'label']]
    result = utils.aggDfCols(cosineDf, allTsCols, func=aggType)
    if saveDir is not None:
        result.to_pickle(saveDir)
        print(f'Results saved as {saveDir}.')
    return result