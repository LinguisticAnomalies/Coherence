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
from coherencecalculator.tools.ppl import PerplexityGenerator
from coherencecalculator.tools.speechgraph import SpeechGraph



import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import warnings

from nltk.tokenize import sent_tokenize

def timeseries(vecLoader:VecLoader, inputText=None, inputDir=None, inputCsv=None, inputPickle=None, inputDf=None, fileCol=None, textCol=None, vecType=None, saveDir=None, embeddingFunc=None, window_size=64, window_batch_size=16) -> pd.DataFrame:
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
    pbar2.close()
    

    # add ppl columns
    pg = PerplexityGenerator(vecLoader=vecLoader)
    cosineDf['avg_ppl'] = cosineDf['sliding_window'] = cosineDf['sliding_window_batch'] = cosineDf['contextmodel'] = cosineDf['topicmodel'] = None
    pbar3 = tqdm(total=len(cosineDf))
    with utils.suppress_stdout():
        for i, row in cosineDf.iterrows():
            text = row['text']
            if type(text) == list:
                freetext = ' '.join(text)
                sentences = text
            else:
                freetext = text
                sentences = sent_tokenize(text)
            avg_ppl, slide_wd_ppl = pg.sliding_window_perplexity(freetext, window_size=window_size, window_batch_size=1)
            avg_ppl, slide_wd_ppl_batch = pg.sliding_window_perplexity(freetext, window_size=window_size, window_batch_size=window_batch_size)
            contextmodel, topicmodel = pg.sentence_level_perplexity(sentences)
            cosineDf.at[i, 'avg_ppl'] = [avg_ppl]
            cosineDf.at[i, 'sliding_window'] = slide_wd_ppl
            cosineDf.at[i, 'sliding_window_batch'] = slide_wd_ppl_batch
            cosineDf.at[i, 'contextmodel'] = contextmodel
            cosineDf.at[i, 'topicmodel'] = topicmodel
            pbar3.update(1)
    pbar3.close()
    
    # add speechgraph columns
    sg = SpeechGraph(vecLoader=vecLoader)
    cosineDf = sg.add_speechgraph_features(cosineDf)
    
    
    allTsCols = [col for col in cosineDf.columns if col not in ['file', 'text', 'label']]
    numDropped = utils.countEmptyTs(cosineDf, allTsCols)
    
    print('Cosine values created.')
    if numDropped > 0:
        print(f'{numDropped} file(s) contain empty timeseries on at least one metric because they are not long enough to produce any cosine values.')
    if saveDir is not None:
        cosineDf.to_pickle(saveDir)
        print(f'Results saved as {saveDir}.')
    return cosineDf

