from __future__ import annotations
from simpletransformers.t5 import T5Model, T5Args
import pandas as pd
import torch

from models.abstract_classifier import TextBinaryClassifier



class T5Wrapper(TextBinaryClassifier):
    def __init__(self, load_pretrained: bool, args: T5Args):
        use_cuda = torch.cuda.is_available()

        if load_pretrained:
            self.model = T5Model(
                args.model_type, args.output_dir,
                args=args, use_cuda=use_cuda
            )
        else:
            self.model = T5Model(
                args.model_type, 't5-base',
                args=args, use_cuda=use_cuda
            )

    def train(self, data: pd.DataFrame):
        torch.cuda.empty_cache()
        self.model.train_model(
            train_data=data,
            show_running_loss=True
        )

    def eval(self, data: pd.DataFrame):
        torch.cuda.empty_cache()
        self.model.eval_model(
            eval_data=data
        )

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy()
        data['prefix'] = 'binary classification'
        data['input_text'] = data['text']
        data['target_text'] = ''
        for i, label in enumerate(data['sarcasm_label']):
            data.loc[i, 'target_text'] = '1' if label == 'sarcastic' else '0'

        return data[['prefix', 'input_text', 'target_text']]
