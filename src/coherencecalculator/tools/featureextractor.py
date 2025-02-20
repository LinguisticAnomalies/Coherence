import pandas as pd
import numpy as np

import warnings
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm.auto import tqdm
class FeatureExtractor(object):

    def __init__(self, pbar) -> None:
        self.pbar = pbar
    
    def __tardis(self, data:pd.DataFrame, idCol:str, tsCol:str) -> pd.DataFrame:
        
        timeseries = pd.DataFrame(columns=['id','time','cos'])
        for index, row in data.iterrows():
            coh = row[tsCol]
            if len(coh) == 0:
                ids = [row[idCol]]
                time = [0]
                coh = [np.nan]
            else:    
                time = np.arange(len(coh))
                ids = [row[idCol]]*len(coh)
            tempdf = pd.DataFrame()
            tempdf['id'] = ids
            tempdf['time'] = time
            tempdf['cos'] = coh
            timeseries = pd.concat([timeseries, tempdf], ignore_index = True)
        #display(timeseries.groupby('id').mean())
        
        rmlist = ['has_duplicate_max',
            'has_duplicate_min',
            'has_duplicate',
            'sum_values',
            'abs_energy',
            'length',
            'absolute_sum_of_changes',
            'count_above_mean',
            'count_below_mean',
            'sum_of_reoccurring_values',
            'sum_of_reoccurring_data_points',
            'ratio_value_number_to_time_series_length',
            'number_cwt_peaks',
            'number_peaks',
            'value_count',
            'range_count']
    
        settings = ComprehensiveFCParameters()
        [settings.pop(key) for key in rmlist]
        features = extract_features(timeseries, column_id = 'id', column_sort='time', default_fc_parameters=settings, disable_progressbar=True)
        features = features.replace([np.inf, -np.inf], np.nan)
        impute(features)
        features[idCol] = features.index
        features.reset_index(inplace=True, drop=True)
        return features
    
    def extractFeatures(self, inputData:pd.DataFrame, idCol:str, methodCols:list) -> dict:
        warnings.filterwarnings("ignore")
        out = {}
        for col in methodCols:
            featuredData = self.__tardis(inputData, idCol, col)
            out[col] = pd.merge(left=featuredData, right=inputData, left_on=idCol, right_on=idCol)
            self.pbar.update(1)
        return out
            