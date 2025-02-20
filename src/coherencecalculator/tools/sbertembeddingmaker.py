from coherencecalculator.tools.vecloader import VecLoader
from coherencecalculator.tools.embeddingmaker import EmbeddingMaker
import coherencecalculator.tools.utils as utils
from nltk.tokenize import sent_tokenize

import numpy as np
import pandas as pd


class SBertEmbeddingMaker(EmbeddingMaker):
    def __init__(self, vecLoader:VecLoader, pbar):
        self.sentBert = vecLoader.sentBert
        self.simCse = vecLoader.simCse
        self.diffCse = vecLoader.diffCse
        self.pbar = pbar

    
    def __normVec(self, vec):
        size = np.linalg.norm(vec)
        if size != 0:
            return vec/size
        else:
            return None

    def __sentCoherenceSentBert(self, incomingData):
        sentences=incomingData
        embeddingList = []
        #modelSentBert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        for sent in sentences:
            sentVec = self.sentBert.encode(sent, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            nSentVec = self.__normVec(sentVec)
            if nSentVec is not None: 
                embeddingList.append(nSentVec)
        return embeddingList

    def __sentCoherenceDiffCSE(self, incomingData):
        sentences=incomingData
        embeddingList = []
        #modelDiffCSE = DiffCSE("voidism/diffcse-bert-base-uncased-sts",device='cuda')
        with utils.suppress_stdout():
            for sent in sentences:
                
                sentVec = self.diffCse.encode(sent,return_numpy=True, normalize_to_unit=True)
                nSentVec = self.__normVec(sentVec)
                if nSentVec is not None: 
                    embeddingList.append(nSentVec)
        return embeddingList

    def __sentCoherenceSimCSE(self, incomingData):
        sentences=incomingData
        embeddingList = []
        #modelSimCSE = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased",device='cuda')
        with utils.suppress_stdout():
            for sent in sentences:
                
                sentVec = self.simCse.encode(sent, return_numpy=True, normalize_to_unit=True)
                nSentVec = self.__normVec(sentVec)
                if nSentVec is not None: 
                    embeddingList.append(nSentVec)
        return embeddingList

    def getEmbeddings(self, inputData:pd.DataFrame) -> pd.DataFrame:
        resultDf = inputData.copy()
        resultDf['sentCoherenceSentBert'] = pd.Series(dtype=object)
        resultDf['sentCoherenceSimCSE'] = pd.Series(dtype=object)
        resultDf['sentCoherenceDiffCSE'] = pd.Series(dtype=object)

        for i, row in resultDf.iterrows():
            if type(row['text']) == list:
                free_text = ' '.join(row['text'])
                segmented = row['text']
            elif type(row['text']) == str:
                free_text = row['text']
                segmented = sent_tokenize(row['text'])
            resultDf.at[i, 'sentCoherenceSentBert'] = self.__sentCoherenceSentBert(segmented)
            resultDf.at[i, 'sentCoherenceSimCSE'] = self.__sentCoherenceSimCSE(segmented)
            resultDf.at[i, 'sentCoherenceDiffCSE'] = self.__sentCoherenceDiffCSE(segmented)
        # resultDf['sentCoherenceSentBert'] = resultDf['text'].apply(self.__sentCoherenceSentBert)
        # resultDf['sentCoherenceSimCSE'] = resultDf['text'].apply(self.__sentCoherenceSimCSE)
        # resultDf['sentCoherenceDiffCSE'] = resultDf['text'].apply(self.__sentCoherenceDiffCSE)
            self.pbar.update(1)
        return resultDf