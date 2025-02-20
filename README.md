## Introduction
This site provides code used to estimate a range of coherence metrics, including sequential (e.g. sentence to subsequent sentence) and global (e.g. sentence to all other sentences) metrics. For further details on the metrics concerned, please see:

Xu W, Wang W, Portanova J, Chander A, Campbell A, Pakhomov S, Ben-Zeev D, Cohen T. Fully Automated Detection of Formal Thought Disorder with Time-series Augmented Representations for Detection of Incoherent Speech (TARDIS). Journal of Biomedical Informatics. 2022 Jan 19:103998.

Xu W, Portanova J, Chander A, Ben-Zeev D, Cohen T. The Centroid Cannot Hold: Comparing Sequential and Global Estimates of Coherence as Indicators of Formal Thought Disorder. InAMIA Annual Symposium Proceedings 2020 (Vol. 2020, p. 1315). American Medical Informatics Association.

## Installation
Note: Updated to support CUDA > 12.0 and Python > 3.12

1. Create new conda environment with python version 3.12.7
```
conda create -n envname python=3.12.7
conda activate envname
```
2. Install the package with pip
```
pip install https://github.com/WeizheXu/coherencepip/releases/download/v0.2.0/coherencecalculator-0.2.0-py3-none-any.whl --no-cache-dir
```
or if you don't have Nvidia GPU, you can use the no cuda version that only uses Word2Vec-like embeddings.
```
pip install https://github.com/WeizheXu/coherencepip/releases/download/v0.1.23_nocuda/coherencecalculatornocuda-0.1.23-py3-none-any.whl --no-cache-dir --no-binary gensim
```
## Basic Use Examples
1. Imports:
```
from coherencecalculator.pipelines.timeseries import timeseries
from coherencecalculator.pipelines.features import features
from coherencecalculator.pipelines.tardis import tardis
from coherencecalculator.pipelines.agg import agg
from coherencecalculator.tools.vecloader import VecLoader
import pandas as pd
```
2. Preload the pretrained embeddings:
```
vecs = VecLoader()
```
3. Create the timeseries data:
```
inputData = '/path/to/data.csv'
tsDf = timeseries(vecLoader=vecs, inputCsv=inputData, fileCol='id', textCol='text')

# or
inputData = '/path/to/data.pkl' # (A pickled pd.DataFrame object)
tsDf = timeseries(vecLoader=vecs, inputPickle=inputData, fileCol='id', textCol='text')

# or 
inputData = pd.DataFrame({'id':[1,2,3], 'text':['a','b','c']})
tsDf = timeseries(vecLoader=vecs, inputDf=inputData, fileCol='id', textCol='text')

# or
inputData = '/path/to/directory' # (A directory contains a list of .txt files)
tsDf = timeseries(vecLoader=vecs, inputDir=inputData)

# or
# sample texts
inputData = 'The memo materials have been infiltrated. They’re jumping around. I used to be good at the broad jump, because I’m tall. I fall. People put things in and then say it’s my fault. I used to be God, but I got demoted.'
tsDf = timeseries(vecLoader=vecs, inputText=inputData)
```
4. Generate aggregation coherence scores (Optional):
```
aggResult = agg(vecLoader=vecs, inputTimeseries=tsDf)
```
5. Generate features from timeseries:
```
featureDict = features(vecLoader=vecs, inputTimeseries=tsDf)
```
6. Generate TARDIS coherence scores:
```
tardisResult = tardis(vecLoader=vecs, inputFeatures=featureDict)
```
7. Train your own models：
Input the target variable column name when creating tsDf
```
tsDf = timeseries(vecLoader=vecs, inputDf=inputData, fileCol='id', textCol='text', labelCol='label')
```
Train model with the feature dictionary
```
featureDict = features(vecLoader=vecs, inputTimeseries=tsDf)
from coherencecalculator.tools.modeltrainer import ModelTrainer
mt = ModelTrainer(featureDict)
mt.saveModel(modelSavePath = '/modelpath.pickle', scalerSavePath='/scalerpath.pickle')
```
To use trained model
```
tardisResult = tardis(vecLoader=vecs, inputFeatures=featureDictTest, modelFile='/modelpath.pickle', scalerFile='/scalerpath.pickle')
```
## Score Interpretation
For tardisResults: higher scores = more incoherent (possible scores: 0-4) These are produced by ML models trained on the TALD scale. 

For aggResults: higher scores = more coherent (possible scoeres: -1-1) These are cosine similarities acquired from embeddings. 
