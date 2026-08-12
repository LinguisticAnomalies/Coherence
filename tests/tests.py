# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import os
from coherencecalculator.pipelines.timeseries import timeseries
from coherencecalculator.pipelines.features import features
from coherencecalculator.pipelines.tardis import tardis
from coherencecalculator.pipelines.agg import agg
from coherencecalculator.tools.vecloader import VecLoader
import pandas as pd


if __name__ == '__main__':
    testsDir = os.path.dirname(os.path.abspath(__file__))
    inputDir = os.path.join(testsDir, 'testData')
    if not os.path.isdir(inputDir):
        raise FileNotFoundError(f'Input data directory not found: {inputDir}')
    vecs = VecLoader()
    outputDir = os.path.join(testsDir, 'output')
    os.makedirs(outputDir, exist_ok=True)
    #If data is csv/pkl/dataframe, file column name and text column name need to be added
    #For example:
    # tsDf = timeseries(vecLoader=vecs, inputCsv=path_to_csv, fileCol='IDCOLNAME', textCol='TEXTCOLNAME', saveDir='./timeseriesout.pkl')
    tsDf = timeseries(vecLoader=vecs, inputDir=inputDir, saveDir=os.path.join(outputDir, 'timeseriesout.pkl'))
    print(pd.read_pickle(os.path.join(outputDir, 'timeseriesout.pkl')))

    featureDict = features(vecLoader=vecs, inputTimeseries=tsDf, saveDir=os.path.join(outputDir, 'featuresout.pkl'))
    print(featureDict)

    tardis(vecLoader=vecs, inputFeatures=featureDict, saveDir=os.path.join(outputDir, 'tardisout.pkl'))
    print(pd.read_pickle(os.path.join(outputDir, 'tardisout.pkl')))

    agg(vecLoader=vecs, inputTimeseries=tsDf, saveDir=os.path.join(outputDir, 'aggout.pkl'))
    print(pd.read_pickle(os.path.join(outputDir, 'aggout.pkl')))