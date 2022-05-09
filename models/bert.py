import pandas as pd
from simpletransformers.classification import ClassificationModel

from models.abstract_classifier import TextBinaryClassifier


class BertWrapper(TextBinaryClassifier):
    def __init__(self, model: ClassificationModel):
        self.model: ClassificationModel = model

    def train(self, data):
        self.model.train_model(
            data,
            show_running_loss=True
        )

    def eval(self, raw_data):
        pass

    def predict(self, text):
        pass

    def prepare_data(self, raw_data: pd.DataFrame):
        data = raw_data.copy()
        # 'text' column already present, no need to do anything
        for i, label in enumerate(data['sarcasm_label']):
            data.loc[i, 'labels'] = 1 if label == 'sarcastic' else 0

        data = data.astype({'text': str, 'labels': int})
        return data[['text', 'labels']]
