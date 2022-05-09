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
            show_running_loss=True
        )

    def eval(self, data: pd.DataFrame):
        self.model.eval_model(
            eval_data=data
        )

    def predict(self, text: str):
        return self.model.predict(text)

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy()
        data['prefix'] = 'binary classification'
        data['input_text'] = data['text']
        for i, label in enumerate(data['sarcasm_label']):
            data.loc[i, 'target_text'] = '1' if label == 'sarcastic' else '0'

        data = data.astype({'prefix': str, 'input_text': str, 'target_text': str})
        return data[['prefix', 'input_text', 'target_text']]
