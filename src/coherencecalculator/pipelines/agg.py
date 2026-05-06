from coherencecalculator.pipelines.timeseries import timeseries
import coherencecalculator.tools.utils as utils
import numpy as np
import pandas as pd
from coherencecalculator.tools.vecloader import VecLoader

def agg(vecLoader:VecLoader, inputTimeseries=None, inputText=None, inputDir=None, inputCsv=None, inputPickle=None, inputDf=None, fileCol=None, textCol=None, vecType=None, saveDir=None, aggType=None) -> pd.DataFrame:
    if inputTimeseries is None:
        cosineDf = timeseries(vecLoader, inputText=inputText, inputDir=inputDir, inputCsv=inputCsv, inputPickle=inputPickle, inputDf=inputDf, fileCol=fileCol, textCol=textCol, vecType=vecType, saveDir=None)
    else:
        cosineDf = inputTimeseries.copy()
    
    allTsCols = [col for col in cosineDf.columns if col not in ['file', 'text', 'label']]
    
    # pplCols = ['avg_ppl', 'sliding_window', 'sliding_window_batch', 'contextmodel', 'topicmodel']
    # minCols = [col for col in allTsCols if col not in pplCols]
    maxCols = ['avg_ppl', 'sliding_window', 'sliding_window_batch']
    medianCols = ['contextmodel', 'topicmodel']
    default_map = {}
    for col in allTsCols:
        if col in maxCols:
            default_map[col] = np.max
        elif col in medianCols:
            default_map[col] = np.median
        else:
            default_map[col] = np.min
    
    if aggType is None:
        aggType = default_map
    
    result = utils.aggDfCols(cosineDf, func_map=aggType)
    
    if saveDir is not None:
        result.to_pickle(saveDir)
        print(f'Results saved as {saveDir}.')
    return result