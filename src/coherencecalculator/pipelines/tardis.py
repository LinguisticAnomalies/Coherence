from coherencecalculator.pipelines.timeseries import timeseries
from coherencecalculator.pipelines.features import features
from coherencecalculator.tools.mlprocessing import MLProcessing
# from coherencecalculator.tools.featureextractor import FeatureExtractor
import coherencecalculator.tools.utils as utils
from coherencecalculator.tools.vecloader import VecLoader

import pandas as pd
from coherencecalculator.tools.assets import MODEL, SCALER, get_asset
def tardis(vecLoader:VecLoader, inputFeatures=None, inputText=None, inputDir=None, inputCsv=None, inputPickle=None, inputDf=None, fileCol=None, textCol=None, vecType=None, saveDir=None) -> pd.DataFrame:
    if inputFeatures is None:
        cosineDf = timeseries(vecLoader, inputText=inputText, inputDir=inputDir, inputCsv=inputCsv, inputPickle=inputPickle, inputDf=inputDf, fileCol=fileCol, textCol=textCol, vecType=vecType, saveDir=None)
        featureDict = features(vecLoader, inputTimeseries=cosineDf)
    else:
        featureDict = inputFeatures
    #features --> coherence scores
    print('Finalizing coherence scores...')
    with utils.suppress_stdout():
        ml = MLProcessing(modelFile=get_asset(MODEL), scalerFile=get_asset(SCALER))
        result = ml.generatePrediction(featureDict)
    if saveDir is not None:
        result.to_pickle(saveDir)
        print(f'Results saved as {saveDir}.')
    
    return result
