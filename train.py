import pandas as pd
import torch
from simpletransformers.t5 import T5Model
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
    previous_model_path = f"{consts.T5_OUTPUT}-{number_of_rows - consts.STEP}"
    if os.path.exists(previous_model_path):
        print(f"\tLoading model from: {previous_model_path}")
        print(f"\tTraining for train_size: {number_of_rows}")
        load_from = previous_model_path
    else:
        print(f"\tTraining for the first time for train_size={number_of_rows}")
        load_from = consts.T5_MODEL_NAME

    t5_args = consts.t5_args
    t5_args.output_dir = f"{consts.T5_OUTPUT}-{number_of_rows}"
    t5 = T5Model(
        t5_args.model_type,
        load_from,
        args=t5_args,
        use_cuda=use_cuda
    )

    return T5Wrapper(t5)


def prepare_bert() -> BertWrapper:
    pass


if __name__ == '__main__':
    utils.seed_torch()

    raw_train_data = pd.read_csv(consts.TRAIN_DATA)
    data_len = len(raw_train_data)

    train_size = consts.INIT_TRAIN_SIZE
    data = raw_train_data[0: train_size]
    while train_size <= consts.MAX_TRAIN_SIZE:
        print("=" * 100)
        print(f"Training for nrows={train_size}")

        model = prepare_t5(train_size)
        trainer = OurTrainer(model)
        trainer.train(data)

        train_size += consts.STEP
        data = raw_train_data[train_size - consts.STEP: train_size]  # next batch of training rows
