import numpy as np
import pandas as pd
import coherencecalculator.tools.utils as utils

class CosineGenerator(object):
    def __init__(self, data, columns, pbar) -> None:
        self.data = data
        self.columns = columns
        self.pbar = pbar
    
    # compute cosine sequentially
    def __sequential(self, vecList):
        seqcohs=[]
        if len(vecList) > 1:
            lastvec=vecList[0]
            for vec in vecList[1:]:
                seqcohs.append(np.dot(vec,lastvec))
                lastvec=vec
        return seqcohs
    
    # compute cosine from the centroid vector, 
    def __staticCentroid(self, vecList):
        fromean=[]
        meanvec=np.sum(vecList,axis=0)
        meanvec=meanvec/np.linalg.norm(meanvec)
        for vec in vecList:
            fromean.append(np.dot(vec,meanvec))
 
        return fromean
    
    # compute cosine from a cummulative centorid
    def __cummulativeCentroid(self, vecList):
        fromrunningmean=[]
        if len(vecList) > 0:
            runningmean = np.zeros(len(vecList[0]))
            runningsum = np.zeros(len(vecList[0]))
            for vec in vecList:
                runningsum = np.add(runningsum, vec)
                runningmean = runningsum/np.linalg.norm(runningsum)#(index+1)
                fromrunningmean.append(np.dot(vec, runningmean))

        return fromrunningmean

    def generateCosine(self) -> pd.DataFrame:
        resultDf = self.data.copy()
        for col in self.columns:
            seq = col + 'Seq'
            sc = col + 'StaticCentroid'
            cc = col + 'CumulativeCentroid'

            resultDf[seq] = pd.Series(dtype=object)
            resultDf[sc] = pd.Series(dtype=object)
            resultDf[cc] = pd.Series(dtype=object)
            for i, row in resultDf.iterrows():
                resultDf.at[i, seq] = self.__sequential(row[col])
                resultDf.at[i, sc] = self.__staticCentroid(row[col])
                resultDf.at[i, cc] = self.__cummulativeCentroid(row[col])
                self.pbar.update(1)
            # resultDf[seq] = resultDf[col].apply(self.__sequential)
            # resultDf[sc] = resultDf[col].apply(self.__staticCentroid)
            # resultDf[cc] = resultDf[col].apply(self.__cummulativeCentroid)
        resultDf = resultDf.drop(columns=self.columns)
        return resultDf
    
    # # secondary coherence 
    # def skip(self):
    #     gapcohs=[]
    #     if len(self.vecList) > 2:
    #         for index, vec in enumerate(self.vecList):
    #             if index+2 < len(self.vecList):
    #                 gapcohs.append(np.dot(vec, self.vecList[index+2]))

    #     return gapcohs
    
    # # cosines from the first vector
    # def fromFirst(self):
    #     fromfirst=[]
    #     for vec in self.vecList:
    #         fromfirst.append(np.dot(vec,self.vecList[0]))

    #     return fromfirst

