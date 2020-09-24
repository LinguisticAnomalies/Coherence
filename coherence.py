#!/usr/bin/env python
# coding: utf-8

import sys
import warnings
sys.path.append('./semvecpy/semvecpy/vectors')
sys.path.append('./semvecpy/semvecpy')
sys.path.append('./spg')
import os
import io
import semvec_utils as sv
import vectors.real_vectors as rv
from platform import python_version
print("python version: " + str(python_version()))
from os import listdir
from os.path import isfile,join
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from textblob import TextBlob as tb
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from scipy.stats import linregress
from scipy.stats import pearsonr
import fileinput
import re
import gensim
from gensim.models import Word2Vec
import gensim.models.keyedvectors as kv
import spg.speechgraph.speechgraph as sg
import networkx as nx
import spacy
nlp = spacy.load("en_core_web_sm")

#option is "naive", "stem" or "partial"
def generategraph(textfile, option):
    if option == "naive":
        graph = sg.naiveGraph() # create a Naive graph
    elif option == "stem":
        graph = sg.stemGraph() # create a Stem Graph
    elif option == "partial":
        graph = sg.posGraph() # create a Part of Speech Graph
    else:
        graph = sg.naiveGraph()
    
    sample = open(textfile, 'rb')
    s = sample.read().decode('utf8', 'ignore')
    result = graph._text2graph(s)
    sample.close()
    return result
    
def getGraphInfoFromText(text, option):
    if option == "naive":
        graph = sg.naiveGraph() # create a Naive graph
    elif option == "stem":
        graph = sg.stemGraph() # create a Stem Graph
    elif option == "partial":
        graph = sg.posGraph() # create a Part of Speech Graph
    elif option == "nostopwords":
        graph = sg.nostopwordGraph()
    else:
        graph = sg.naiveGraph()
    result = graph.analyzeText(text)
    return result

def model2dict(model):
    term_dict = {}

    for word in model.wv.vocab:
        term_dict[word] = model.wv.__getitem__(word)

    #normalize term_dict vectors
    for key in term_dict:
        term_dict[key] = term_dict[key]/np.linalg.norm(term_dict[key])

    return term_dict

def maken2vmodel(textfile, namedotmodel, dimension, walklength, numberwalks):
    testgraph = generategraph(textfile, "naive")
    print('creating model from graph')
    n2v = Node2Vec(testgraph, dimensions=dimension, walk_length=walklength, num_walks=numberwalks)
    model = n2v.fit(window=10, min_count=1)
    print('model created')
    model.save(namedotmodel)
    print('model saved')

def normalize(series):
    nseries = []
    series_nonan = series.dropna()
    minimum = series_nonan.min()
    maximum = series_nonan.max()
    for x in series:
        if np.isnan(x):
            nseries.append(x)
        else:
            nx = (x-minimum)/(maximum-minimum)
            nseries.append(nx)
    nseries = pd.Series(nseries)
    return nseries

def get_idf(word, idfdict):
    if word in idfdict:
        return idfdict[word]
    else:
        return 1

def generatecoherence(vectors):
    seqcohs=[]
    fromean=[]
    gapcohs=[]
    fromfirst=[]
    fromrunningmean=[]
    
    #seq vectors
    if len(vectors) > 1:
        lastvec=vectors[0]
        for vec in vectors[1:]:
            seqcohs.append(np.dot(vec,lastvec))
            lastvec=vec
    
    #mean vectors
    meanvec=np.sum(vectors,axis=0)
    meanvec=meanvec/np.linalg.norm(meanvec)
    for vec in vectors:
        fromean.append(np.dot(vec,meanvec))
    
    #running mean vectors
    if len(vectors) > 0:
        runningmean = np.zeros(len(vectors[0]))
        runningsum = np.zeros(len(vectors[0]))
        for index, vec in enumerate(vectors):
            runningsum = np.add(runningsum, vec)
            runningmean = runningsum/np.linalg.norm(runningsum)#(index+1)
            fromrunningmean.append(np.dot(vec, runningmean))
            
    #Gap
    if len(vectors) > 2:
        for index, vec in enumerate(vectors):
            if index+2 < len(vectors):
                gapcohs.append(np.dot(vec, vectors[index+2]))
    
    #from first vectors
    for vec in vectors:
        fromfirst.append(np.dot(vec,vectors[0]))
    
    #from all vectors
    fromall=np.array(vectors)
    fromall=np.dot(fromall, fromall.T)
    
    return [seqcohs, fromean, fromrunningmean, gapcohs, fromfirst, fromall]

def wordcoherence(term_dict,incomingdata):
    stop_words = set(stopwords.words('english')) 
    terms = word_tokenize(incomingdata)
    terms = [w for w in terms if not w.lower() in stop_words] 

    localwordvectors=[]
    
    for term in terms:
        term=term.lower()
        if term in term_dict:
            v0=term_dict[term]
            localwordvectors.append(v0)
    
    #return localwordvectors
    return generatecoherence(localwordvectors)

def removeDupFromList(x):
    res = [] 
    for i in x: 
        if i not in res: 
            res.append(i)
    return res

def removeConsecDup(x):
    previous_value = None
    new_lst = []

    for elem in x:
       if elem != previous_value:
           new_lst.append(elem)
           previous_value = elem
    return new_lst

def wordcoherenceUnique(term_dict,incomingdata):
    stop_words = set(stopwords.words('english')) 
    terms = word_tokenize(incomingdata)
    terms = [w for w in terms if not w.lower() in stop_words]
    terms = removeConsecDup(terms)
    localwordvectors=[]
    
    for term in terms:
        term=term.lower()
        if term in term_dict:
            v0=term_dict[term]
            localwordvectors.append(v0)
    
    #return localwordvectors
    return generatecoherence(localwordvectors)


def sentcoherenceweighted(term_dict,incomingdata,idf_dict):
    ps=sent_tokenize(incomingdata)
    stop_words = set(stopwords.words('english')) 
    sentvecs=[]
    
    for sent in ps:
        sentvec=np.zeros(len(term_dict[next(iter(term_dict))]))
        terms=word_tokenize(sent)
        terms= [w for w in terms if not w.lower() in stop_words] 
        for term in terms:
                    if term.lower() in term_dict:
                        weight = get_idf(term.lower(), idf_dict)
                        v1=term_dict[term.lower()]*weight
                        sentvec=np.add(v1,sentvec)
       
        if np.linalg.norm(sentvec) > 0:
            sentvec=sentvec/np.linalg.norm(sentvec)
            sentvecs.append(sentvec)
    
    return generatecoherence(sentvecs)

def sumdictvalues(freq_dict):
    N = 0
    for key, value in freq_dict.items():
        N = N + value
    return N

def wordunigramprob(sumfreq, word, freq_dict):
    if word in freq_dict:
        return_val = freq_dict[word]/sumfreq
    else:
        return_val = 1
    return return_val

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def sentcoherencesif(term_dict, incomingdata, sumfreq, freq_dict, a=1e-3, npc=1):
    ps=sent_tokenize(incomingdata)
    stop_words = set(stopwords.words('english')) 
    sentvecs=[]
    #sentarray=np.array()
    
    for index, sent in enumerate(ps):
        sentvec=np.zeros(len(term_dict[next(iter(term_dict))]))
        terms=word_tokenize(sent)
        terms= [w for w in terms if not w.lower() in stop_words] 
        for term in terms:
                    if term.lower() in term_dict:
                        weight = a/(a+wordunigramprob(sumfreq, term.lower(), freq_dict))
                        v1=term_dict[term.lower()]*weight
                        sentvec=np.add(v1,sentvec)
        
        if np.linalg.norm(sentvec) > 0:
            sentvec=sentvec/np.linalg.norm(sentvec)
        sentvecs.append(sentvec)
        
    #return generatecoherence(sentvecs)
        if index == 0:
            sentarray = np.array(sentvec)
        else:
            sentarray = np.column_stack((sentarray, np.array(sentvec)))
    if sentarray.ndim > 1:
        sentarray = remove_pc(sentarray, npc)
        sentarray = sentarray.transpose()
        return_val = sentarray.tolist()
    else:
        sentarray = sentarray.transpose()
        return_val = [sentarray.tolist()]

    
    return generatecoherence(return_val)
    
   
def sentcoherence(term_dict,incomingdata):
    ps=sent_tokenize(incomingdata)
    stop_words = set(stopwords.words('english')) 
    sentvecs=[]
    
    for sent in ps:
        sentvec=np.zeros(len(term_dict[next(iter(term_dict))]))
        terms=word_tokenize(sent)
        terms= [w for w in terms if not w.lower() in stop_words] 
        for term in terms:
                    if term.lower() in term_dict:
                        v1=term_dict[term.lower()]
                        sentvec=np.add(v1,sentvec)
       
        if np.linalg.norm(sentvec) > 0:
            sentvec=sentvec/np.linalg.norm(sentvec)
            sentvecs.append(sentvec)

    return generatecoherence(sentvecs)

def paircoherence(term_dict, incomingdata):
    stop_words = set(stopwords.words('english')) 
    terms = word_tokenize(incomingdata)
    terms = [w for w in terms if not w.lower() in stop_words]
    pairvecs = []
    for index, term in enumerate(terms):
        if index % 2 != 0:
            continue
        if index+1 < len(terms):
            pairvec = np.zeros(len(term_dict[next(iter(term_dict))]))
            #Add the two words together or keep one if the other word cannot be found in dictionary
            if term.lower() in term_dict:
                tempv = term_dict[term.lower()]
                pairvec = np.add(tempv, pairvec)
            if terms[index+1].lower() in term_dict:
                tempv = term_dict[terms[index+1].lower()]
                pairvec = np.add(tempv, pairvec)
            if np.linalg.norm(pairvec) > 0:
                pairvec = pairvec/np.linalg.norm(pairvec)
                pairvecs.append(pairvec)
    return generatecoherence(pairvecs)

def gapcoherence(term_dict, incomingdata):
    stop_words = set(stopwords.words('english')) 
    terms = word_tokenize(incomingdata)
    terms = [w for w in terms if not w.lower() in stop_words]
    gapvecs = []
    
    for index, term in enumerate(terms):
        if index % 2 != 0:
            continue
        if term.lower() in term_dict:
            tempv = term_dict[term.lower()]
            gapvecs.append(tempv)
    return generatecoherence(gapvecs)

def phrasecoherence(term_dict, incomingdata):
    blob = tb(incomingdata)
    phrases = blob.noun_phrases
    # doc = nlp(incomingdata)
    # phrases = []
    # for chunk in doc.noun_chunks:
    #     phrases.append(chunk.text)
    
    
    stop_words = set(stopwords.words('english')) 
    
    phrasevecs=[]
    
    for phrase in phrases:
        phrasevec=np.zeros(len(term_dict[next(iter(term_dict))]))
        terms=word_tokenize(phrase)
        terms= [w for w in terms if not w.lower() in stop_words] 
        for term in terms:
                    if term.lower() in term_dict:
                        v1=term_dict[term.lower()]
                        phrasevec=np.add(v1,phrasevec)
       
        if np.linalg.norm(phrasevec) > 0:
            phrasevec=phrasevec/np.linalg.norm(phrasevec)
            phrasevecs.append(phrasevec)
    
    return generatecoherence(phrasevecs)


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

def gettermdict(option):
    if option.endswith('.bin'):
        termvectors=sv.readfile(option)
        termstore = rv.RealVectorStore()
        termstore.init_from_lists(termvectors[0],termvectors[1])
        termstore.normalize_all()
        term_dict=dict(zip(termvectors[0],termvectors[1]))
        return term_dict
    elif option.endswith('.model'):
        model_skipgram_loaded = Word2Vec.load(option)
        term_dict = {}
        for word in model_skipgram_loaded.wv.vocab:
            term_dict[word] = model_skipgram_loaded.wv.__getitem__(word)
        #normalize term_dict vectors
        for key in term_dict:
            term_dict[key] = term_dict[key]/np.linalg.norm(term_dict[key])
        return term_dict
    elif option.endswith('.vec'):
        term_dict = load_vectors(option)
        for key in term_dict:
            term_dict[key] = term_dict[key]/np.linalg.norm(term_dict[key])
        return term_dict
    else: 
        return {}

def populatedf(df, term_dict, freq_dict, idf_dict, func):
    N=sumdictvalues(freq_dict)
    for index, row in df.iterrows(): 
        wordc = wordcoherence(term_dict, row['text'])
        phrazc = phrasecoherence(term_dict, row['text'])
        sentc = sentcoherence(term_dict, row['text'])
        wordUnique = wordcoherenceUnique(term_dict, row['text'])
        #gapc = gapcoherence(term_dict, row['text'])
        sentc_weighted = sentcoherenceweighted(term_dict, row['text'], idf_dict)
        #sent_sif = sentcoherencesif(term_dict, row['text'], N)
        
        units = ['word', 'phrase', 'sent', 'sentidf']
        methods = ['seq','mean','runningmean','gap']
        
        
        try:
            df.at[index, 'wordseq'] = func(wordc[0])
            #df.at[index, 'wordsequnique'] = func(wordUnique[0])
            df.at[index, 'wordmean'] = func(wordc[1])
            #df.at[index, 'wordmeanunique']=func(wordUnique[1])
            df.at[index, 'wordrunningmean'] = func(wordc[2])
            #df.at[index, 'wordrunningmeanunique'] = func(wordUnique[2])
            df.at[index, 'wordgap'] = func(wordc[3])
        except ValueError:
            df.at[index, 'wordseq'] = np.nan
            df.at[index, 'wordmean'] = np.nan
            #df.at[index, 'wordsequnique'] = np.nan
            df.at[index, 'wordrunningmean'] = np.nan
            df.at[index, 'wordgap'] = np.nan

        try:        
            df.at[index, 'phraseseq'] = func(phrazc[0])
            df.at[index, 'phrasemean'] = func(phrazc[1])
            df.at[index, 'phraserunningmean'] = func(phrazc[2])
            #df.at[index, 'phrasegap'] = func(phrazc[3])
        except ValueError:
            df.at[index, 'phraseseq'] = np.nan
            df.at[index, 'phrasemean'] = np.nan
            df.at[index, 'phraserunningmean'] = np.nan
            #df.at[index, 'phrasegap'] = np.nan


        try:        
            df.at[index, 'sentseq'] = func(sentc[0])
            df.at[index, 'sentmean'] = func(sentc[1])
            df.at[index, 'sentrunningmean'] = func(sentc[2])
            #df.at[index, 'sentgap'] = func(sentc[3])
        except ValueError: 
            df.at[index, 'sentseq'] = np.nan
            df.at[index, 'sentmean'] = np.nan
            df.at[index, 'sentrunningmean'] = np.nan
            #df.at[index, 'sentgap'] = np.nan

        try:        
            df.at[index, 'sentidfseq'] = func(sentc_weighted[0])
            df.at[index, 'sentidfmean'] = func(sentc_weighted[1])
            df.at[index, 'sentidfrunningmean'] = func(sentc_weighted[2])
            #df.at[index, 'sentidfgap'] = func(sentc_weighted[3])
        except ValueError: 
            df.at[index, 'sentidfseq'] = np.nan
            df.at[index, 'sentidfmean'] = np.nan
            df.at[index, 'sentidfrunningmean'] = np.nan
            #df.at[index, 'sentidfgap'] = np.nan


        filename = row['file']
        try:
            graphstats = getGraphInfoFromText(row['text'], 'naive')

        except:
            df.at[index, 'number_of_nodes'] = np.nan
            df.at[index, 'number_of_edges'] = np.nan
            df.at[index, 'number_of_parallel_edges'] = np.nan
            df.at[index, 'largest_strongly_connected_component'] = np.nan
            df.at[index, 'average_degree'] = np.nan
            df.at[index, 'loop1node'] = np.nan
            df.at[index, 'loop2node'] = np.nan
            df.at[index, 'loop3node'] = np.nan
            df.at[index, 'density'] = np.nan
            df.at[index, 'diameter'] = np.nan
            df.at[index, 'asp'] = np.nan
        else:
            df.at[index, 'number_of_nodes'] = graphstats['number_of_nodes']
            df.at[index, 'number_of_edges'] = graphstats['number_of_edges']
            df.at[index, 'number_of_parallel_edges'] = graphstats['PE']
            df.at[index, 'largest_strongly_connected_component'] = graphstats['LSC']
            df.at[index, 'average_degree'] = graphstats['degree_average']
            df.at[index, 'loop1node'] = graphstats['L1']
            df.at[index, 'loop2node'] = graphstats['L2']
            df.at[index, 'loop3node'] = graphstats['L3']
            df.at[index, 'density'] = graphstats['density']
            df.at[index, 'diameter'] = graphstats['diameter']
            df.at[index, 'asp'] = graphstats['asp']

    df['normal_number_of_nodes'] = (1-normalize(df['number_of_nodes']))
    df['normal_number_of_edges'] = (1-normalize(df['number_of_edges']))
    df['normal_number_of_parallel_edges'] = (1-normalize(df['number_of_parallel_edges']))
    df['normal_largest_strongly_connected_component'] = (normalize(df['largest_strongly_connected_component']))
    df['normal_average_degree'] = (1-normalize(df['average_degree']))
    df['normal_loop1node'] = (1-normalize(df['loop1node']))
    df['normal_loop2node'] = (1-normalize(df['loop2node']))
    df['normal_loop3node'] = (1-normalize(df['loop3node']))
    df['normal_density'] = normalize(df['density'])
    df['normal_diameter'] = normalize(df['diameter'])
    df['normal_asp'] = normalize(df['asp'])

#'fasttext_vectors.bin', 'r5_basic_embeddingvectors.bin', 'lemmatizedwiki.model', 'nolemmawiki.model'
def generatetable(vectorspace, df, freq_dict, idf_dict, func):
    term_dict = gettermdict(vectorspace)
    newdf = df.copy()
    populatedf(newdf, term_dict, freq_dict, idf_dict, func)
    tocompare = [
    'wordseq',
    'wordmean',
    'wordrunningmean',
    'wordgap',
    'phraseseq',
    'phrasemean',
    'phraserunningmean',
    'sentseq',
    'sentmean',
    'sentrunningmean',
    'sentidfseq',
    'sentidfmean',
    'sentidfrunningmean'
    ]
    for t in tocompare:
        newdf[t] = normalize(newdf[t])
    return newdf

def gettxts(directory):
    all_files = os.listdir(directory)
    txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
    filenames = []
    txts = []
    for f in txt_files:
        with open(directory + f, 'rt', encoding='ISO-8859-1') as fd:
            filenames.append(f)
            txts.append(fd.read())
    autotransdf = pd.DataFrame()
    autotransdf['file'] = filenames
    autotransdf['text'] = txts
    return autotransdf

def main():
    warnings.filterwarnings("ignore")
    if '-p' not in sys.argv:
        print('Please specify data directory where .txt files are stored')
        print("Using e.g. python3 coherence.py -p './data/directory'")
        return
    downloadpath = './coherenceresult.csv'
    for index, arg in enumerate(sys.argv):
        if arg == '-p':
            if index + 1 >= len(sys.argv):
                print('Please specify data directory where .txt files are stored')
                print("Using e.g. python3 coherence.py -p './data/directory/'") 
                return
            elif str(sys.argv[index+1]).startswith('-'):
                print('Please specify data directory where .txt files are stored')
                print("Using e.g. python3 coherence.py -p './data/directory/'") 
                return         
            else:
                data_dir = sys.argv[index+1]
        if arg == '-s':
            if index + 1 >= len(sys.argv):
                print('To save the results, use -s directory')
                print("e.g. coherence.py -p './datadir/' -s '~/Downloads/results.csv'")
                return
            elif str(sys.argv[index+1]).startswith('-'):
                print('To save the results, use -s directory')
                print("e.g. coherence.py -p './datadir/' -s '~/Downloads/results.csv'")
                return 
            else:
                downloadpath = sys.argv[index+1]
        if arg == '-f':
            if index + 1 >= len(sys.argv):
                print('Please specify min mean median max after -f. If not, the default aggregation function is min')
                aggreFunc = np.min
                return
            elif str(sys.argv[index+1]).startswith('-'):
                print('Please specify min mean median max after -f. If not, the default aggregation function is min')
                aggreFunc = np.min
                return
            elif sys.argv[index+1] == 'min':
                aggreFunc = np.min
            elif sys.argv[index+1] == 'mean':
                aggreFunc = np.mean
            elif sys.argv[index+1] == 'median':
                aggreFunc = np.median
            elif sys.argv[index+1] == 'max':
                aggreFunc = np.max
            else:
                aggreFunc = np.min
        if arg == '-v':
            if index + 1 >= len(sys.argv):
                print('Please specify vectors after -v. If not, the default fasttext vectors trained from common crawl without subword')
                aggreFunc = np.min
                return
            elif str(sys.argv[index+1]).startswith('-'):
                print('Please specify vectors after -v. If not, the default fasttext vectors trained from common crawl without subword')
                aggreFunc = np.min
                return
            else:
                vecs = str(sys.argv[index+1])
    if '-f' not in sys.argv:
        print('Aggregation function is set to default: minimum')
        aggreFunc = np.min
    if '-v' not in sys.argv:
        print('Vector is set to default: fasttext_vectors.bin')
        vecs = 'fasttext_vectors.bin'


    try:
        textData = gettxts(data_dir)
    except:
        print(sys.exc_info()[0])
        return
    print('Generating results. This usually takes several minutes')
#     print(textData.head())
    idf=pd.read_csv('./wikiidf_terms.csv')
    idf_dict = idf.set_index('token')['idf'].to_dict()
    freq_dict = idf.set_index('token')['frequency'].to_dict()
    #fastext = generatetable('crawl-300d-2M-subword.vec', textData, freq_dict, idf_dict)
    fastext = generatetable(vecs, textData, freq_dict, idf_dict, aggreFunc)
    #fastext = fastext.dropna()
    print('Result preview:')
    print(fastext.head())
    fastext.to_csv(downloadpath, index = None, header=True)
    print("Results saved as csv. If you did not specify a path using -s, the default is './coherenceresult.csv'")


if __name__ == "__main__":
    main()
