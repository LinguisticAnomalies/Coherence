# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import sys
sys.path.append('/edata/coherencenotebook/coherencecalculator/src')
from coherencecalculator.pipelines.timeseries import timeseries
from coherencecalculator.pipelines.features import features
from coherencecalculator.pipelines.tardis import tardis
from coherencecalculator.pipelines.agg import agg
from coherencecalculator.tools.vecloader import VecLoader
import pandas as pd
import pickle

if __name__ == '__main__':
    inputDir = '/edata/coherencenotebook/coherencecalculator/tests/testData'
    vecs = VecLoader()
    #If data is csv/pkl/dataframe, file column name and text column name need to be added
    #For example: 
    # tsDf = timeseries(vecLoader=vecs, inputCsv=path_to_csv, fileCol='IDCOLNAME', textCol='TEXTCOLNAME', saveDir='./timeseriesout.pkl')
    tsDf = timeseries(vecLoader=vecs, inputDir=inputDir, saveDir='./timeseriesout.pkl')
    print(pd.read_pickle('./timeseriesout.pkl'))
    
    featureDict = features(vecLoader=vecs, inputTimeseries=tsDf, saveDir='./featuresout.pkl')
    print(featureDict)
    
    tardis(vecLoader=vecs, inputFeatures=featureDict, saveDir='./tardisout.pkl')
    print(pd.read_pickle('./tardisout.pkl'))
    
    agg(vecLoader=vecs, inputTimeseries=tsDf, saveDir='./aggout.pkl')
    print(pd.read_pickle('./aggout.pkl'))