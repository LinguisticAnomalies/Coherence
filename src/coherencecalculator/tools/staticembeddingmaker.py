from coherencecalculator.tools.embeddingmaker import EmbeddingMaker

from coherencecalculator.tools.vecloader import VecLoader

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import spacy
nlp = spacy.load("en_core_web_sm")

import numpy as np
import pandas as pd



class StaticEmbeddingMaker(EmbeddingMaker):
    def __init__(self, vecLoader:VecLoader, pbar):
        
        self.term_dict = vecLoader.term_dict
        self.idfDict = vecLoader.idfDict
        self.pbar = pbar
    
    def __wordCoherence(self, incomingdata) -> list:
        stop_words = set(stopwords.words('english')) 
        terms = word_tokenize(incomingdata)
        terms = [w for w in terms if not w.lower() in stop_words]

        localwordvectors=[]
        
        for term in terms:
            term=term.lower()
            if term in self.term_dict:
                v0=self.term_dict[term]
                localwordvectors.append(v0)
        return localwordvectors
    
    def __phraseCoherence(self, incomingdata) -> list:
        
        doc = nlp(incomingdata)
        phrases = []
        for chunk in doc.noun_chunks:
            phrases.append(chunk.text)
        
        stop_words = set(stopwords.words('english')) 
        
        phrasevecs=[]
        
        for phrase in phrases:
            phrasevec=np.zeros(len(self.term_dict[next(iter(self.term_dict))]))
            terms=word_tokenize(phrase)
            terms= [w for w in terms if not w.lower() in stop_words]

            for term in terms:
                if term.lower() in self.term_dict:
                    v1=self.term_dict[term.lower()]
                    phrasevec=np.add(v1,phrasevec)
        
            if np.linalg.norm(phrasevec) > 0:
                phrasevec=phrasevec/np.linalg.norm(phrasevec)
                phrasevecs.append(phrasevec)
        
        return phrasevecs

    def __sentCoherence(self, incomingdata, weighted=False) -> list:
        ps = incomingdata
        stop_words = set(stopwords.words('english')) 
        sentvecs=[]
        
        for sent in ps:
            sentvec=np.zeros(len(self.term_dict[next(iter(self.term_dict))]))
            terms=word_tokenize(sent)
            terms= [w for w in terms if not w.lower() in stop_words]

            for term in terms:
                if term.lower() in self.term_dict:
                    if weighted:
                        if term.lower() in self.idfDict:
                            weight = self.idfDict[term.lower()]
                        else:
                            weight = 1
                        v1=self.term_dict[term.lower()]*weight
                    else:
                        v1=self.term_dict[term.lower()]
                    
                    sentvec=np.add(v1,sentvec)
        
            if np.linalg.norm(sentvec) > 0:
                sentvec=sentvec/np.linalg.norm(sentvec)
                sentvecs.append(sentvec)
        return sentvecs
    
    def getEmbeddings(self, inputData:pd.DataFrame) -> pd.DataFrame:
        resultDf = inputData.copy()
        resultDf['wordCoherence'] = pd.Series(dtype='object')
        resultDf['phraseCoherence'] = pd.Series(dtype='object')
        resultDf['sentCoherence'] = pd.Series(dtype='object')
        resultDf['sentCoherenceWeighted'] = pd.Series(dtype='object')
        for i, row in resultDf.iterrows():
            if type(row['text']) == list:
                free_text = ' '.join(row['text'])
                segmented = row['text']
            elif type(row['text']) == str:
                free_text = row['text']
                segmented = sent_tokenize(row['text'])
            resultDf.at[i, 'wordCoherence'] = self.__wordCoherence(free_text)
            resultDf.at[i, 'phraseCoherence'] = self.__phraseCoherence(free_text)
            resultDf.at[i, 'sentCoherence'] = self.__sentCoherence(segmented)
            resultDf.at[i, 'sentCoherenceWeighted'] = self.__sentCoherence(segmented, weighted=True)
            self.pbar.update(1)
        # resultDf['wordCoherence'] = resultDf['text'].apply(self.__wordCoherence)
        # resultDf['phraseCoherence'] = resultDf['text'].apply(self.__phraseCoherence)
        # resultDf['sentCoherence'] = resultDf['text'].apply(self.__sentCoherence)
        # resultDf['sentCoherenceWeighted'] = resultDf['text'].apply(self.__sentCoherence, args=(True,))
        return resultDf
