import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm

class MLProcessing(object):
    def __init__(self, modelFile:str, scalerFile:str, modelType:str='reg') -> None:
        # modelType in ['reg', 'clf']
        self.allModels = pickle.load(open(modelFile, 'rb'))
        self.allScalers = pickle.load(open(scalerFile, 'rb'))
        self.modelType = modelType

    def __pred(self, data:pd.DataFrame, tsCol:str):
        data1 = data.copy()
        scalers = self.allScalers[tsCol]
        model = self.allModels[tsCol]
        
        featureCols = [key for key in scalers.keys() if key in data1.columns]
        for col in featureCols:
            scaler = scalers[col]
            toScale = np.reshape(np.array(data1[col]), (-1,1))
            data1[col] = scaler.transform(toScale)
        
        if len(featureCols) == 1:
            x_test = data1[featureCols].values.reshape(-1, 1)
        else:
            x_test = data1[featureCols].values

        if self.modelType == 'reg':
            return model.predict(x_test)
        else:
            return model.predict_proba(x_test)[:,1]
    
    def generatePrediction(self, featureDict:dict) -> pd.DataFrame:
        methodCols = list(featureDict.keys())

        for i, col in enumerate(tqdm(methodCols)):
            # keep a copy of the original data by dropping the feature columns
            if i == 0:
                scalers = self.allScalers[col]
                features = featureDict[col]
                featureCols = [key for key in scalers.keys() if key in features.columns]
                resultDf = features.drop(columns=featureCols)
                
            resultDf[col] = self.__pred(featureDict[col], col)
        return resultDf