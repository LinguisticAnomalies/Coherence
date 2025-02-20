## Comprehensive Coherence Calculator

This site provides code used to estimate a range of coherence metrics, including sequential (e.g. sentence to subsequent sentence) and global (e.g. sentence to all other sentences) metrics. For further details on the metrics concerned, please see:

Xu W, Wang W, Portanova J, Chander A, Campbell A, Pakhomov S, Ben-Zeev D, Cohen T. Fully Automated Detection of Formal Thought Disorder with Time-series Augmented Representations for Detection of Incoherent Speech (TARDIS). Journal of Biomedical Informatics. 2022 Jan 19:103998.

Xu W, Portanova J, Chander A, Ben-Zeev D, Cohen T. The Centroid Cannot Hold: Comparing Sequential and Global Estimates of Coherence as Indicators of Formal Thought Disorder. InAMIA Annual Symposium Proceedings 2020 (Vol. 2020, p. 1315). American Medical Informatics Association.

## Download docker image
`docker pull xuweizhe/comp_coh_calc_nogpu` for no gpu setup. \
`docker pull xuweizhe/comp_coh_calc_gpu` for gpu setup.
## Start docker container from image
Go to data directory and then: \
``docker run --name cohContainer -v `pwd`:`pwd` -it xuweizhe/comp_coh_calc_nogpu`` \
If using the gpu version then: \
``docker run --gpus all --name cohContainer -v `pwd`:`pwd` -it xuweizhe/comp_coh_calc_gpu``
## Inside container
1. `conda activate coherence`
2. `python3 coherenceMain.py --dataDir='/path/to/directory/with/txts'`

### Options for coherenceMain.py:
1. `--config`: select from [`'fasttext'`, `'gensim'`, `'bert'`] (default: `'fasttext'`)
2. `--mode`: (default: `sentence`) \
if `--config` is not `'bert'`: \
select from [`'word'`, `'phrase'`, `'sentence'`, `'sentenceIdf'`] . \
if `--config` is `'bert'`: \
select from [`'word'`, `'phrase'`, `'sentence'`, `'sentenceCls'`, `'sentence2ndLayer'`, `'sentenceSBert'`]
3. `--cosineType`: select from [`'sequential'`, `'staticCentroid'`, `'cumulativeCentroid'`] (default: `'staticCentroid'`)
4. `--featureType`: select from [`'agg'`, `'tardis'`] (default: `'tardis'`)
5. `--aggFunc`: select from [`np.min`, `np.max`, `np.mean`, `np.median`, `np.std`] (default: `np.min`)
6. `--preTrained`: (default: `'fasttext_vectors.bin'`) \
if `--config` is `'fasttext'`: \
Use a 'vectors.bin' file. \
if `--config` is `'gensim'`: \
Use a gensim pre-trained Word2Vec model. (e.g. `"glove-wiki-gigaword-50"`)
7. `--dataDir`: Directory containing all txt files. (No default)
8. `--outDir`: Directory and csv file name to save the results to. (default: `'./coherenceResults.csv'`)
## Citation:

1. Mota NB, Vasconcelos NAP, Lemos N, Pieretti AC, Kinouchi O, Cecchi GA, et al. (2012) Speech Graphs Provide a Quantitative Measure of Thought Disorder in Psychosis. PLoS ONE 7(4): e34928. https://doi.org/10.1371/journal.pone.0034928 Github page: https://github.com/guillermodoghel/speechgraph
2. Semvecpy software package. Github page: https://github.com/semanticvectors/semvecpy
3. Fasttext software package. Url:https://fasttext.cc/docs/en/english-vectors.html


