from contextlib import contextmanager
import sys, os
import pandas as pd
import numpy as np

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def dropEmptyTs(df:pd.DataFrame, columns:list) -> pd.DataFrame:
    toDrop = []

    for i, row in df.iterrows():
        for col in columns:
            if len(row[col]) == 0:
                toDrop.append(i)
                break
    out = df.drop(index=toDrop)
    return out.reset_index(drop=True)

def countEmptyTs(df:pd.DataFrame, columns:list) -> pd.DataFrame:
    toDrop = []

    for i, row in df.iterrows():
        for col in columns:
            if len(row[col]) == 0:
                toDrop.append(i)
                break
    
    return len(toDrop)

def aggDfCols(df:pd.DataFrame, func_map:dict) -> pd.DataFrame:
    result = df.copy()
    for col, func in func_map.items():
        if col in result.columns:
            for i, row in result.iterrows():
                if len(row[col]) > 0:
                    result.at[i, col] = func(row[col])
                else:
                    result.at[i, col] = np.nan
    return result