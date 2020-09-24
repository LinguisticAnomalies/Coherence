## 1. File preparation:
1. Clone this directory.
2. Download pre-trained vectors from https://fasttext.cc/docs/en/english-vectors.html
3. Download IDF data file from https://drive.google.com/file/d/1Xw2wyx6P8yit--FxB7a2NwmFXLPxJaIT/view?usp=sharing
4. Download semvecpy package to load semantic vectors. `git clone https://github.com/semanticvectors/semvecpy.git`
## 2. Prepare conda environment:
1. Create environment: `conda create -n coherence python=3.7`
2. Activate environment: `conda activate coherence`
3. Install dependencies: 
   * numpy: `conda install numpy`
   * bitarray: `conda install -c anaconda bitarray`
   * scipy: `conda install scipy`
   * pandas: `conda install pandas`
   * nltk: `conda install nltk`(nltk.download('stopwords'))
   * textblob: `conda install -c conda-forge textblob`
   * scikitlearn: `conda install -c anaconda scikit-learn`
   * gensim: `conda install gensim`
   * networkx: `conda install networkx`
   * multiprocess: `conda install -c conda-forge multiprocess`
   * numba: `conda install numba`
   * spacy: `conda install -c conda-forge spacy`
## 3. Run the python file:
1. Execute file: `python3 coherence.py -p './data_dir/'`
(Please don't forget the '/' in the end)
2. The result will be saved as coherenceresult.csv in current directory.
3. If you want to save it to somewhere else, use -s directory.
e.g. `python3 coherence.py -p './data_dir/' -s '~/coherenceresult.csv'`
(Please also specify the filename in the end as the example shows)
4. Use '-f' tag to specify aggregation function used to summarize the coherence of a document.
The available options are 'min' 'mean' 'max' and 'median'. e.g. -f mean. Default is min. 
## 4. Interprete the results:
<p>This csv results contains several metrics of the coherence of the input text files. Some of them are based on 
word embedding vectors and some of them are based on graph analysis. Below is a short description of the meaning of 
the column names you get from the csv.</p>

1. Vector based measures:
   <p>These measures come from the cosine values of embedding vectors. They should produce a value between 0-1 with 0 being
   not coherent. Many measures have the names composed of two parts, like "wordmin" (word + min), 
   they can be interpreted as such:</p>

   1. word/sent/phraz/gap 
      * Word = every word as unit
      * Sent = sentence as unit
      * Phraz = only includes noun phrases
      * Gap = only count every other word
   2. min/mean/runningmean
      * min = the cosine is calculated based on consecutive units.
      * mean = the cosine is calculated based on each unit and the mean value. 
      * runningmean = the cosine is calculated based on the running mean value.
   3. some sentence measures have "weighted" or "sif"
      * weighted = the vectors in each word of the sentence are weighted by inverse document frequency (IDF)
      * sif = the sentence vectors are calculated using smooth inverse frequency (SIF) (The numbers may not look right for these
      ones because I just added this measure and I need to make sure I have the calculations right.
2. Graph based measures.
   <p>These measures are based on putting the words on a graph data structure and analyze the property of the graph. 
   These measures should all start with 'normal' as I normalized them to a number between 0-1 so it can be compared to previous
   measures.</p>
## 5. Citation:

1. Mota NB, Vasconcelos NAP, Lemos N, Pieretti AC, Kinouchi O, Cecchi GA, et al. (2012) Speech Graphs Provide a Quantitative Measure of Thought Disorder in Psychosis. PLoS ONE 7(4): e34928. https://doi.org/10.1371/journal.pone.0034928 Github page: https://github.com/guillermodoghel/speechgraph
2. Semvecpy software package. Github page: https://github.com/semanticvectors/semvecpy
3. Fasttext software package. Url:https://fasttext.cc/docs/en/english-vectors.html


