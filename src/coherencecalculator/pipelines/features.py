from coherencecalculator.pipelines.timeseries import timeseries
import coherencecalculator.tools.utils as utils
from coherencecalculator.tools.featureextractor import FeatureExtractor
import pickle
from coherencecalculator.tools.vecloader import VecLoader
from tqdm.auto import tqdm

def features(vecLoader:VecLoader, inputTimeseries=None, inputText = None, inputDir=None, inputCsv=None, inputPickle=None, inputDf=None, fileCol=None, textCol=None, vecType=None, saveDir=None) -> dict:
    if inputTimeseries is None:
        cosineDf = timeseries(vecLoader, inputText=inputText, inputDir=inputDir, inputCsv=inputCsv, inputPickle=inputPickle, inputDf=inputDf, fileCol=fileCol, textCol=textCol, vecType=vecType, saveDir=None)
    else:
        cosineDf = inputTimeseries.copy()
    
    allTsCols = [col for col in cosineDf.columns if col not in ['file', 'text', 'label']]
    #cosines --> features
    print('Generating features...')
    pbar = tqdm(total=len(allTsCols))
    with utils.suppress_stdout():
        fe = FeatureExtractor(pbar)
        featureDict = fe.extractFeatures(cosineDf, 'file', allTsCols)
    print('Features created.')
    if saveDir is not None:
        pickle.dump(featureDict, open(saveDir, 'wb'))
        print(f'Results saved as {saveDir}.')
    return featureDict