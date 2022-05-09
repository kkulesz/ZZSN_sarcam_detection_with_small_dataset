import pandas as pd

from models.abstract_classifier import TextBinaryClassifier


class OurTrainer:
    def __init__(self, model: TextBinaryClassifier):
        self.model = model

    def train(self, raw_data: pd.DataFrame):
        data = self.model.prepare_data(raw_data)
        self.model.train(data)

    def eval(self, raw_data: pd.DataFrame):
        data = self.model.prepare_data(raw_data)
        self.model.eval(data)
