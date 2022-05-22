import pandas as pd
from simpletransformers.t5 import T5Model
from simpletransformers.classification import ClassificationModel
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.bert_sarcasm_classifier import BertSarcasmClasifierWrapper
from models.t5 import T5Wrapper
from models.bert import BertWrapper
import consts
import utils

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


def prepare_bert(number_of_rows: int) -> BertWrapper:
    bert_args = consts.BERT_ARGS
    bert_args.output_dir = f"{consts.BERT_OUTPUT}-{number_of_rows}"
    if consts.CURRENT_TASK == consts.TASK_DETECT:
        bert = ClassificationModel(
            consts.BERT_MODEL_TYPE,
            consts.BERT_MODEL_NAME,
            args=bert_args,
            use_cuda=use_cuda
        )
        return BertWrapper(bert)
    else:
        bert = ClassificationModel(
            consts.BERT_MODEL_TYPE,
            consts.BERT_MODEL_NAME,
            args=bert_args,
            use_cuda=use_cuda,
            num_labels=6
        )
        return BertSarcasmClasifierWrapper(bert)



if __name__ == '__main__':
    utils.prepare_environment()

    data = pd.read_csv(consts.TEST_TESTING_DATA)
    if consts.CURRENT_TASK == consts.TASK_DETECT:
        inputs, labels = utils.prepare_evaluation_data(data)
    else:
        inputs, labels = utils.prepare_sarcasm_evaluation_data(data)

    train_size = consts.INIT_TRAIN_SIZE
    while train_size <= consts.MAX_TRAIN_SIZE:
        torch.cuda.empty_cache()
        print("="*100)
        print(f"Evaluating {consts.CURRENT_VARIANT} for nrows={train_size}")

        if consts.CURRENT_VARIANT == consts.T5:
            model = prepare_t5(train_size)
        elif consts.CURRENT_VARIANT == consts.BERT:
            model = prepare_bert(train_size)
        else:
            raise Exception(f"Invalid variant={consts.CURRENT_VARIANT}!")

        predictions = model.predict(inputs)
        print(predictions)
        print(f" -Number of predicted sarcasms: {sum(predictions)}\n")

        print(f" -Precision:  {precision_score(labels, predictions, average='micro')}")
        print(f" -Accuracy:   {accuracy_score(labels, predictions)}")
        print(f" -Recall:     {recall_score(labels, predictions, average='weighted')}")
        print(f" -F1 score:   {f1_score(labels, predictions, average='macro')}")

        train_size += consts.STEP
