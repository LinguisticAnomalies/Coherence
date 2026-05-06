from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

class ModelTrainer(object):
    def __init__(self, featureDict:dict, labelCol='label'):
        self.featureDict = featureDict
        self.labelCol = labelCol

    def __tardisTrain(self, trainDf):

        scalers = {}
        featureCols = [c for c in trainDf.columns if c.startswith('cos__')]
        for col in featureCols:
            scaler = MinMaxScaler(clip=True)
            toScaleTrain = np.reshape(np.array(trainDf[col]), (-1,1))
            trainDf[col] = scaler.fit_transform(toScaleTrain)
            scalers[col] = scaler

        y_train = trainDf[self.labelCol].values
        if len(featureCols) == 1:
            x_train = trainDf[featureCols].values.reshape(-1, 1)
        else:
            x_train = trainDf[featureCols].values
        
        model = SVR(kernel='rbf')
        reg = model.fit(x_train, y_train)
        return scalers, reg

    def __trainModels(self):
        allTsCols = list(self.featureDict.keys())
        scalers = {}
        models = {}
        for ts in allTsCols:
            trainDf = self.featureDict[ts].copy()
            scaler, reg = self.__tardisTrain(trainDf)
            scalers[ts] = scaler
            models[ts] = reg
        return scalers, models
            
    def saveModel(self, modelSavePath:str, scalerSavePath:str):
        allScalers, allModels = self.__trainModels()
        pickle.dump(allModels, open(modelSavePath, 'wb'))
        pickle.dump(allScalers, open(scalerSavePath, 'wb'))