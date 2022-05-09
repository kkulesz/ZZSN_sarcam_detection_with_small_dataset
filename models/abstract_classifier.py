from abc import ABC, abstractmethod
import pandas as pd


class TextBinaryClassifier(ABC):
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def eval(self, data):
        pass

    @abstractmethod
    def predict(self, text):
        pass
