# disable progressbar in the imported packages
import tqdm
def nop(it, *a, **k):
    return it
tqdm.tqdm = nop

from coherencecalculator.tools.datacreator import DataCreator
from coherencecalculator.tools.staticembeddingmaker import StaticEmbeddingMaker
from coherencecalculator.tools.bertembeddingmaker import BertEmbeddingMaker
from coherencecalculator.tools.sbertembeddingmaker import SBertEmbeddingMaker
from coherencecalculator.tools.cosinegenerator import CosineGenerator
import coherencecalculator.tools.utils as utils
from coherencecalculator.tools.vecloader import VecLoader

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import warnings

def timeseries(vecLoader:VecLoader, inputText=None, inputDir=None, inputCsv=None, inputPickle=None, inputDf=None, fileCol=None, textCol=None, vecType=None, saveDir=None, embeddingFunc=None) -> pd.DataFrame:
    #read in data
    print('Loading data...')
    dc = DataCreator()
    if inputText is not None:
        data = dc.fromText(inputText)
    elif inputDf is not None:
        data = dc.fromDf(inputDf, fileCol=fileCol, textCol=textCol)
    elif inputDir is not None:
        data = dc.fromDir(inputDir)
    elif inputCsv is not None:
        data = dc.fromCsv(inputCsv, fileCol=fileCol, textCol=textCol)
    elif inputPickle is not None:
        data = dc.fromPickle(inputPickle, fileCol=fileCol, textCol=textCol)
    else:
        print('Please specify input data with --inputDir, --inputCsv or --inputPickle')
        return
    print('Loading data completed.')
    
    
    #text --> embeddings
    print('Generating embeddings...')
    if vecType == 'static':
        pbar1 = tqdm(total=len(data))
        with utils.suppress_stdout():
            stEmb = StaticEmbeddingMaker(vecLoader, pbar1)
            embeddings = stEmb.getEmbeddings(data)
    elif vecType == 'bert':
        pbar1 = tqdm(total=len(data))
        with utils.suppress_stdout():
            bertEm = BertEmbeddingMaker(vecLoader, pbar1)
            embeddings = bertEm.getEmbeddings(data)
    elif vecType == 'sbert':
        pbar1 = tqdm(total=len(data))
        with utils.suppress_stdout():
            sBertEm = SBertEmbeddingMaker(vecLoader, pbar1)
            embeddings = sBertEm.getEmbeddings(data)
    elif vecType == 'custom':
        pbar1 = tqdm(total=len(data))
        embeddings = data.copy()
        embeddings['custom_embedding'] = pd.Series(dtype=object)
        for i, row in embeddings.iterrows():
            embeddings.at[i, 'custom_embedding'] = embeddingFunc(row['text'])
            pbar1.update(1)
    else:
        pbar1 = tqdm(total=3*len(data))
        with utils.suppress_stdout():
            stEmb = StaticEmbeddingMaker(vecLoader, pbar1)
            embeddings = stEmb.getEmbeddings(data)
            bertEm = BertEmbeddingMaker(vecLoader, pbar1)
            embeddings = bertEm.getEmbeddings(embeddings)
            sBertEm = SBertEmbeddingMaker(vecLoader, pbar1)
            embeddings = sBertEm.getEmbeddings(embeddings)
            
    pbar1.close()

    #embeddings --> cosines
    print('Computing cosines...')
    allMethods = [col for col in embeddings.columns if col not in ['file', 'text', 'label']]
    pbar2 = tqdm(total=len(allMethods)*len(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        cg = CosineGenerator(data=embeddings, columns=allMethods, pbar=pbar2)
        cosineDf = cg.generateCosine()
    allTsCols = [col for col in cosineDf.columns if col not in ['file', 'text', 'label']]
    cosineDf = utils.dropEmptyTs(cosineDf, allTsCols)
    numDropped = len(data) - len(cosineDf)
    pbar2.close()
    print('Cosine values created.')
    if numDropped > 0:
        print(f'{numDropped} file(s) are dropped because they are not long enough to produce any cosine values.')
    if saveDir is not None:
        cosineDf.to_pickle(saveDir)
        print(f'Results saved as {saveDir}.')
    return cosineDf

