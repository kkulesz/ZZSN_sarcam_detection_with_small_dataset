import pandas as pd
from models.bert import BertWrapper


class BertSarcasmClasifierWrapper(BertWrapper):
    def prepare_training_data(self, raw_data: pd.DataFrame):
        data = raw_data.copy()

        # 'text' column already present, no need to do anything
        data['labels'] = data['sarcasm_type']
        data = data.astype({'text': str, 'labels': int})

        return data[['text', 'labels']]
