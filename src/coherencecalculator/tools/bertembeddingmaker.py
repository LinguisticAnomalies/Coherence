import torch
#from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

from coherencecalculator.tools.embeddingmaker import EmbeddingMaker
from coherencecalculator.tools.vecloader import VecLoader

from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize

import spacy
nlp = spacy.load("en_core_web_sm")

import numpy as np
import pandas as pd

class BertEmbeddingMaker(EmbeddingMaker):
    def __init__(self, vecLoader:VecLoader, pbar) -> None:
        # Load pre-trained model (weights)
        self.model = vecLoader.bertModel
        #self.device = torch.device(device)
        self.device = vecLoader.device
        if vecLoader.useCuda:
            self.model.to(self.device)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        self.useCuda = vecLoader.useCuda

        self.tokenizer = vecLoader.bertTokenizer
        self.pbar = pbar
    
    def __generateEmbeddingDict(self, text, word=False):
        marked_text = "[CLS] " + text + " [SEP]"
        indexed_tokens = self.tokenizer(marked_text, truncation=True).input_ids
        tokenized_text = self.tokenizer.convert_ids_to_tokens(indexed_tokens)
        # tokenized_text = self.tokenizer.tokenize(marked_text)
        # indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        if self.useCuda:
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            segments_tensors = torch.tensor([segments_ids]).to(self.device)
        else:
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]     # `hidden_states` has shape [13 x 1 x n x 768]
        
        if word:
            # sum last 4 layers for word embeddings
            embeddings = hidden_states[-4]
            for tensor in hidden_states[-3:]:
                embeddings = embeddings+tensor #[1 x n x 768]
            embeddings = embeddings[0] #[n x 768]
            embeddingDict = {}
            for i, e in enumerate(embeddings):
                if self.useCuda:
                    embeddingDict[tokenized_text[i]] = e.detach().cpu().numpy()
                else:
                    embeddingDict[tokenized_text[i]] = e.detach().numpy()
            return embeddingDict
        else:
            # `token_vecs` is a tensor with shape [n x 768]
            # 2nd to last layer as sentence embedding
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all n token vectors.
            if self.useCuda:
                sentence_embeddings = torch.mean(token_vecs, dim=0).cpu()
            else:
                sentence_embeddings = torch.mean(token_vecs, dim=0)

            return sentence_embeddings.numpy()

    def __wordCoherenceBertSum(self, incomingData) -> list:
        sentences=incomingData
        stop_words = set(stopwords.words('english')) 
        embeddingList = []
        for sent in sentences:
            vectorDict = self.__generateEmbeddingDict(sent, word=True)
            for token, vec in vectorDict.items():
                if token not in ['[CLS]', '[SEP]'] and token not in stop_words:
                    embeddingList.append(vec/np.linalg.norm(vec))
        return embeddingList

    def __phraseCoherenceBertSum(self, incomingData) -> list:
        sentences=incomingData
        stop_words = set(stopwords.words('english'))
        embeddingList = []
        for sent in sentences:
            vectorDict = self.__generateEmbeddingDict(sent, word=True)
            doc = nlp(sent)
            phrases = []
            for chunk in doc.noun_chunks:
                phrases.append(chunk.text)
            for ph in phrases:
                tokens = self.tokenizer.tokenize(ph)
                sumVec = np.zeros(len(vectorDict['[CLS]']))
                for token in tokens:
                    if token not in ['[CLS]', '[SEP]'] and token not in stop_words:
                        try:
                            vec = vectorDict[token]
                        except KeyError:
                            continue
                        sumVec = sumVec + vec
                vecNorm = np.linalg.norm(sumVec)
                if vecNorm != 0:
                    embeddingList.append(sumVec/vecNorm)
        return embeddingList

    def __sentCoherenceBertSum(self, incomingData) -> list:
        sentences=incomingData
        stop_words = set(stopwords.words('english')) 
        embeddingList = []
        for sent in sentences:
            vectorDict = self.__generateEmbeddingDict(sent, word=True)
            sumVec = np.zeros(len(vectorDict['[CLS]']))
            for token, vec in vectorDict.items():
                if token not in ['[CLS]', '[SEP]'] and token not in stop_words:
                    sumVec = sumVec + vec
            vecNorm = np.linalg.norm(sumVec)
            if vecNorm != 0:
                embeddingList.append(sumVec/vecNorm)
        return embeddingList

    def __sentCoherenceBert2ndLayer(self, incomingData) -> list:
        sentences=incomingData
        embeddingList = []
        for sent in sentences:
            sentVec = self.__generateEmbeddingDict(sent, word=False)
            embeddingList.append(sentVec/np.linalg.norm(sentVec))
        return embeddingList
    
    def __sentCoherenceBertCls(self, incomingData) -> list:
        sentences=incomingData
        embeddingList = []
        for sent in sentences:
            vectorDict = self.__generateEmbeddingDict(sent, word=True)
            embeddingList.append(vectorDict['[CLS]']/np.linalg.norm(vectorDict['[CLS]']))
        return embeddingList


    def getEmbeddings(self, inputData:pd.DataFrame) -> pd.DataFrame:
        resultDf = inputData.copy()
        resultDf['wordCoherenceBertSum'] = pd.Series(dtype=object)
        resultDf['phraseCoherenceBertSum'] = pd.Series(dtype=object)
        resultDf['sentCoherenceBertSum'] = pd.Series(dtype=object)
        resultDf['sentCoherenceBert2ndLayer'] = pd.Series(dtype=object)
        resultDf['sentCoherenceBertCls'] = pd.Series(dtype=object)
        for i, row in resultDf.iterrows():
            if type(row['text']) == list:
                free_text = ' '.join(row['text'])
                segmented = row['text']
            elif type(row['text']) == str:
                free_text = row['text']
                segmented = sent_tokenize(row['text'])
            
            resultDf.at[i, 'wordCoherenceBertSum'] = self.__wordCoherenceBertSum(segmented)
            resultDf.at[i, 'phraseCoherenceBertSum'] = self.__phraseCoherenceBertSum(segmented)
            resultDf.at[i, 'sentCoherenceBertSum'] = self.__sentCoherenceBertSum(segmented)
            resultDf.at[i, 'sentCoherenceBert2ndLayer'] = self.__sentCoherenceBert2ndLayer(segmented)
            resultDf.at[i, 'sentCoherenceBertCls'] = self.__sentCoherenceBertCls(segmented)
            self.pbar.update(1)
        # resultDf['wordCoherenceBertSum'] = resultDf['text'].apply(self.__wordCoherenceBertSum)
        # resultDf['phraseCoherenceBertSum'] = resultDf['text'].apply(self.__phraseCoherenceBertSum)
        # resultDf['sentCoherenceBertSum'] = resultDf['text'].apply(self.__sentCoherenceBertSum)
        # resultDf['sentCoherenceBert2ndLayer'] = resultDf['text'].apply(self.__sentCoherenceBert2ndLayer)
        # resultDf['sentCoherenceBertCls'] = resultDf['text'].apply(self.__sentCoherenceBertCls)
        return resultDf