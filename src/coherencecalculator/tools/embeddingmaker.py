from abc import ABC, abstractmethod
import pandas as pd

class EmbeddingMaker(ABC):
    @abstractmethod
    def getEmbeddings(self, inputData:pd.DataFrame) -> pd.DataFrame:
        pass