import pandas as pd
import os

class DataCreator(object):
    def __init__(self, save=False, dropEmpty=True, emptyThreshold=0) -> None:
        self.saveData = save
        self.dropEmpty = dropEmpty
        self.emptyThreshold = emptyThreshold
    
    def __dropEmpty(self, resultDf:pd.DataFrame):
        toDrop=[]
        for i, row in resultDf.iterrows():
            if len(str(row['text']).split()) <= self.emptyThreshold:
                toDrop.append(i)
        numEmpty = len(toDrop)
        if numEmpty:
            print(f'{numEmpty} samples are automatically dropped because they are empty.')
            return resultDf.drop(index=toDrop).reset_index(drop=True)
        else:
            return resultDf

    def __processDf(self, inputDf, fileCol, textCol, labelCol):
        if labelCol is not None:
            resultDf = inputDf[[fileCol, textCol, labelCol]].copy().rename(columns={fileCol:'file', textCol:'text', labelCol:'label'})
        else:
            resultDf = inputDf[[fileCol, textCol]].copy().rename(columns={fileCol:'file', textCol:'text'})
        if self.dropEmpty:
            resultDf = self.__dropEmpty(resultDf)
        if self.saveData:
            resultDf.to_pickle('./dataDf.pkl')
        return resultDf
    
    def fromDir(self, directory):
        path = directory+'/'
        all_files = os.listdir(path)
        txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
        filenames = []
        txts = []
        for f in txt_files:
            with open(path + f, 'rt', encoding='ISO-8859-1') as fd:
                filenames.append(f)
                txts.append(fd.read())
        autotransdf = pd.DataFrame()
        autotransdf['file'] = filenames
        autotransdf['text'] = txts
        toReturn = self.__processDf(inputDf=autotransdf, fileCol='file', textCol='text', labelCol=None)
        return toReturn
    
    def fromCsv(self, csvFile, fileCol, textCol, labelCol=None):
        inputDf = pd.read_csv(csvFile)
        toReturn = self.__processDf(inputDf=inputDf, fileCol=fileCol, textCol=textCol, labelCol=labelCol)
        return toReturn

    def fromPickle(self, pklFile, fileCol, textCol, labelCol=None):
        inputDf = pd.read_pickle(pklFile)
        toReturn = self.__processDf(inputDf=inputDf, fileCol=fileCol, textCol=textCol, labelCol=labelCol)
        return toReturn
    
    def fromDf(self, df, fileCol, textCol, labelCol=None):
        inputDf = df.copy()
        toReturn = self.__processDf(inputDf=inputDf, fileCol=fileCol, textCol=textCol, labelCol=labelCol)
        return toReturn
    
    def fromText(self, text):
        result = pd.DataFrame()
        result['file'] = ['input_text.txt']
        result['text'] = [text]
        if self.saveData:
            result.to_pickle('./dataDf.pkl')
        return result        

    
        