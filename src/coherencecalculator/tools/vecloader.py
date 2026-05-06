
import pkg_resources
from semvecpy.vectors import real_vectors as rv
import gensim.downloader
import pandas as pd
import numpy as np
from coherencecalculator.diffcse import DiffCSE
from sentence_transformers import SentenceTransformer
from simcse import SimCSE
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy


DATAPATH = pkg_resources.resource_filename('coherencecalculator', 'vecs/')
VECSOURCE = 'fasttext'
VECFILE = DATAPATH + 'fasttext_vectors.bin'
IDFFILE = DATAPATH + 'wikiidf_terms.csv'
MODELCARD = "EleutherAI/pythia-1.4b-deduped"

class VecLoader(object):
    def __init__(self, vecType='fasttext', preTrained=VECFILE, idfData=IDFFILE, useCuda=True, device='cuda:0'):
        print('Loading vectors...')
        if vecType == 'fasttext':
            self.__loadFastText(preTrained)
        elif vecType == 'gensim':
            self.__loadGensim(preTrained)
        
        self.nlp = spacy.load("en_core_web_sm")
        self.__loadIdf(idfData)

        self.bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.bertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
        self.useCuda = useCuda
        self.device = device
        
        self.llm, self.llm_tokenizer = self.__loadLlm(MODELCARD, device=self.device)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)

        if useCuda:
            self.sentBert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.simCse = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased",device=device)
            self.diffCse = DiffCSE("voidism/diffcse-bert-base-uncased-sts",device=device)
        else:
            self.sentBert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.simCse = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased",device='cpu')
            self.diffCse = DiffCSE("voidism/diffcse-bert-base-uncased-sts",device='cpu')
    
    def __loadFastText(self, preTrained) -> None:
        termstore = rv.RealVectorStore()
        termstore.init_from_file(preTrained)
        termstore.normalize_all()
        term_dict=dict(zip(termstore.terms, termstore.vectors))
        self.term_dict = term_dict
    
    def __loadGensim(self, preTrained) -> None:
        model_skipgram_loaded = gensim.downloader.load(preTrained)
        term_dict = {}
        for word in model_skipgram_loaded.wv.vocab:
            term_dict[word] = model_skipgram_loaded.wv.__getitem__(word)
        #normalize term_dict vectors
        for key in term_dict:
            term_dict[key] = term_dict[key]/np.linalg.norm(term_dict[key])
        self.term_dict = term_dict
    
    def __loadIdf(self, idfData) -> None:
        idfDf = pd.read_csv(idfData)
        self.idfDict = idfDf.set_index('token')['idf'].to_dict()
        
    def __loadLlm(self, model_card, device):
        """Setup the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_card)
        model = AutoModelForCausalLM.from_pretrained(model_card)
        
        # Handle special tokens if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = model.to(device)
        model.eval()
        
        return model, tokenizer