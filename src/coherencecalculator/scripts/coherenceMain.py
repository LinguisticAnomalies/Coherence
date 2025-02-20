
import numpy as np
import argparse

from coherencecalculator.tools.datacreator import DataCreator
from coherencecalculator.tools.staticembeddingmaker import StaticEmbeddingMaker
from coherencecalculator.tools.bertembeddingmaker import BertEmbeddingMaker
from coherencecalculator.tools.sbertembeddingmaker import SBertEmbeddingMaker
from coherencecalculator.tools.cosinegenerator import CosineGenerator
from coherencecalculator.tools.featureextractor import FeatureExtractor
from coherencecalculator.tools.mlprocessing import MLProcessing

import coherencecalculator.tools.utils as utils
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    warnings.filterwarnings('ignore')
    vecTypeChoices = ['static', 'bert', 'sbert', 'all']
    staticChoices = ['fasttext', 'gensim']
    modeChoices = ['agg', 'tardis']
    aggChoices = [np.min, np.mean, np.median, np.std]
    
    # vector arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--vecType', type=str, default='all', dest = 'vecType', choices = vecTypeChoices, help='')
    parser.add_argument('--vecSource', type=str, default='fasttext', dest='vecSource', choices=staticChoices, help='')
    parser.add_argument('--vecFile', type=str, default='../vecs/fasttext_vectors.bin', dest = 'vecFile', help='Location to the pretrained vector file.')
    parser.add_argument('--idfFile', type=str, default='../vecs/wikiidf_terms.csv', dest = 'idfFile', help='Location to the idf weight file.')
    # tadis vs min arguments
    parser.add_argument('--mode', type=str, default='tardis', dest='mode', choices=modeChoices, help='')
    tempArgs = parser.parse_known_args()[0]
    if tempArgs.mode == 'agg':
        parser.add_argument('--aggType', default=np.min, dest='aggType', choices=aggChoices, help='')
    elif tempArgs.mode == 'tardis':
        parser.add_argument('--tardisModels', type=str, default='../models/model_original.pickle', dest='tardisModels', help='.pickle file for a dictionary of tardis models.')
        parser.add_argument('--tardisScalers', type=str, default='../models/scaler_original.pickle', dest='tardisScalers', help='.pickle file for a dictionary of tardis scalers.')
    # input output arguments
    parser.add_argument('--dataDir', type=str, default=None, dest='inputDir', help='Input data folder contains txt files.')
    parser.add_argument('--dataCsv', type=str, default=None, dest='inputCsv', help='Input csv file contains transcript name and text columns.')
    parser.add_argument('--dataPkl', type=str, default=None, dest='inputPickle', help='Input pickled dataframe file contains transcript name and text columns')
    parser.add_argument('--fileCol', type=str, dest='fileCol', help='Transcript name column of the csv file.')
    parser.add_argument('--textCol', type=str, dest='textCol', help='Text column of the input csv file.')
    
    parser.add_argument('--outputDir', type=str, dest='outputDir', default='./result.csv', help='CSV file output save directory and file name.')

    args = parser.parse_args()
    
    #read in data
    print('Loading data...')
    dc = DataCreator()
    if args.inputDir is not None:
        data = dc.fromDir(args.inputDir)
    elif args.inputCsv is not None:
        data = dc.fromCsv(args.inputCsv, fileCol=args.fileCol, textCol=args.textCol)
    elif args.inputPickle is not None:
        data = dc.fromPickle(args.inputPickle, fileCol=args.fileCol, textCol=args.textCol)
    else:
        print('Please specify input data with --inputDir, --inputCsv or --inputPickle')
        return
    print('Loading data completed.')
    
    #text --> embeddings
    print('Generating embeddings...')
    with utils.suppress_stdout():
        if args.vecType == 'static':
            stEmb = StaticEmbeddingMaker(vecType=args.vecSource, preTrained=args.vecFile, idfData=args.idfFile)
            embeddings = stEmb.getEmbeddings(data)
        elif args.vecType == 'bert':
            bertEm = BertEmbeddingMaker()
            embeddings = bertEm.getEmbeddings(data)
        elif args.vecType == 'sbert':
            sBertEm = SBertEmbeddingMaker()
            embeddings = sBertEm.getEmbeddings(data)
        else:
            stEmb = StaticEmbeddingMaker(vecType=args.vecSource, preTrained=args.vecFile, idfData=args.idfFile)
            embeddings = stEmb.getEmbeddings(data)
            bertEm = BertEmbeddingMaker()
            embeddings = bertEm.getEmbeddings(embeddings)
            sBertEm = SBertEmbeddingMaker()
            embeddings = sBertEm.getEmbeddings(embeddings)


    #embeddings --> cosines
    print('Computing cosines...')
    allMethods = [col for col in embeddings.columns if col not in ['file', 'text', 'label']]
    cg = CosineGenerator(data=embeddings, columns=allMethods)
    cosineDf = cg.generateCosine()
    allTsCols = [col for col in cosineDf.columns if col not in ['file', 'text', 'label']]
    cosineDf = utils.dropEmptyTs(cosineDf, allTsCols)
    numDropped = len(data) - len(cosineDf)
    print('Cosine values created.')

    if args.mode == 'tardis':
        #cosines --> features
        print('Generating features...')
        with utils.suppress_stdout():
            fe = FeatureExtractor()
            featureDict = fe.extractFeatures(cosineDf, 'file', allTsCols)
        print('Features created.')
        
        #features --> coherence scores
        print('Finalizing coherence scores...')
        with utils.suppress_stdout():
            ml = MLProcessing(modelFile=args.tardisModels, scalerFile=args.tardisScalers)
            result = ml.generatePrediction(cosineDf, featureDict)
    elif args.mode == 'agg':
        print('Finalizing coherence scores...')
        result = utils.aggDfCols(cosineDf, allTsCols, func=args.aggType)
    
    result.to_csv(args.outputDir, index=False)
    numTotal = len(data)
    print(f'Results saved to {args.outputDir}')
    print(f'{numDropped} out of {numTotal} transcripts failed to produce a coherence score. They are dropped from the output file.')

if __name__ == "__main__":
    main()