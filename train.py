import pandas as pd
import torch
from simpletransformers.t5 import T5Model
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import warnings
import os

from models.t5 import T5Wrapper
from models.bert import BertWrapper
from trainer import OurTrainer
import consts
import utils

# ugly, but simpletransfomers.T5 throws some stupid
# deprecation warnings if everything is done the way
# the official tutorial says: https://simpletransformers.ai/docs/t5-model/
warnings.filterwarnings("ignore", category=FutureWarning)

use_cuda = torch.cuda.is_available()


def prepare_t5(number_of_rows: int) -> T5Wrapper:
    t5_args = consts.T5_ARGS
    t5_args.output_dir = f"{consts.T5_OUTPUT}-{number_of_rows}"
    t5 = T5Model(
        t5_args.model_type,
        consts.T5_MODEL_NAME,
        args=t5_args,
        use_cuda=use_cuda
    )

    return T5Wrapper(t5)


def prepare_bert(number_of_rows: int) -> BertWrapper:
    bert_args = consts.BERT_ARGS
    bert_args.output_dir = f"{consts.BERT_OUTPUT}-{number_of_rows}"
    bert = ClassificationModel(
        bert_args.model_type,
        consts.BERT_MODEL_NAME,
        args=bert_args,
        use_cuda=use_cuda
    )

    return BertWrapper(bert)


if __name__ == '__main__':
    utils.seed_torch()

    raw_train_data = pd.read_csv(consts.TRAIN_DATA)
    data_len = len(raw_train_data)

    train_size = consts.INIT_TRAIN_SIZE
    while train_size <= consts.MAX_TRAIN_SIZE:
        torch.cuda.empty_cache()
        print("=" * 100)
        print(f"Training for nrows={train_size}")
        data = raw_train_data[0: train_size]

        # model = prepare_t5(train_size)
        model = prepare_bert(train_size)

        trainer = OurTrainer(model)
        trainer.train(data)

        train_size += consts.STEP
