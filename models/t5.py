from __future__ import annotations
from simpletransformers.t5 import T5Model
import pandas as pd

from models.abstract_classifier import TextBinaryClassifier


class T5Wrapper(TextBinaryClassifier):
    def __init__(self, model: T5Model):
        self.model: T5Model = model

    def train(self, data: pd.DataFrame):
        self.model.train_model(
            train_data=data,
            show_running_loss=False
        )

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        integer_predictions = list(map(lambda p: int(p), predictions))
        return integer_predictions

    def prepare_training_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy()

        data['prefix'] = 'binary classification'
        data['input_text'] = data['text']
        data['target_text'] = data['labels']
        data = data.astype({'prefix': str, 'input_text': str, 'target_text': str})

        return data[['prefix', 'input_text', 'target_text']]
