import pandas as pd
from simpletransformers.t5 import T5Model
from simpletransformers.classification import ClassificationModel
import torch
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.t5 import T5Wrapper
from models.bert import BertWrapper
import consts
import utils

# ugly, but simpletransfomers.T5 throws some stupid
# deprecation warnings if everything is done the way
# the official tutorial says: https://simpletransformers.ai/docs/t5-model/
warnings.filterwarnings("ignore", category=FutureWarning)

use_cuda = torch.cuda.is_available()


def prepare_t5(nrows: int) -> T5Wrapper:
    model_dir = f"{consts.T5_OUTPUT}-{nrows}"
    t5_args = consts.T5_ARGS
    t5_args.output_dir = f"{model_dir}-eval"

    print(f"\t Loading model from: {model_dir}")
    t5 = T5Model(
        t5_args.model_type,
        model_dir,  # load already trained model
        args=t5_args,
        use_cuda=use_cuda
    )

    return T5Wrapper(t5)


def prepare_bert(nrows: int) -> BertWrapper:
    model_dir = f"{consts.BERT_OUTPUT}-{nrows}"
    bert_args = consts.BERT_ARGS
    bert = ClassificationModel(
        bert_args.model_type,
        model_dir,  # load already trained model
        args=bert_args,
        use_cuda=use_cuda
    )

    return BertWrapper(bert)


if __name__ == '__main__':
    utils.seed_torch()

    data = pd.read_csv(consts.TEST_DATA)

    inputs = data['text'].tolist()
    labels = data['sarcasm_label'].map(lambda l: 1 if l == 'sarcastic' else 0).tolist()

    train_size = consts.INIT_TRAIN_SIZE
    while train_size <= consts.MAX_TRAIN_SIZE:
        torch.cuda.empty_cache()
        print("="*100)
        print(f"Evaluating for nrows={train_size}")
        # model = prepare_t5(train_size)
        model = prepare_bert(train_size)

        predictions = model.predict(inputs)
        print(predictions)
        print(f" -Number of predicted sarcasms: {sum(predictions)}\n")

        print(f" -Precision:  {precision_score(labels, predictions)}")
        print(f" -Accuracy:   {accuracy_score(labels, predictions)}")
        print(f" -Recall:     {recall_score(labels, predictions)}")
        print(f" -F1 score:   {f1_score(labels, predictions)}")

        train_size += consts.STEP
