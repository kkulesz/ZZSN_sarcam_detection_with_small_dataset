import os
from simpletransformers.t5 import T5Args
from simpletransformers.classification import ClassificationArgs

T5 = 't5'
BERT = 'bert'
CURRENT_VARIANT = BERT
####################################################################
DATA_DIR = 'data'
TRAIN_DATA = os.path.join(DATA_DIR, 'train_preprocessed.csv')
TEST_DATA = os.path.join(DATA_DIR, 'test_preprocessed.csv')

T5_OUTPUT = 't5-output'
BERT_OUTPUT = 'bert-output'

INIT_TRAIN_SIZE = 500
MAX_TRAIN_SIZE = 2700
STEP = 500
#####################################################################
T5_MODEL_TYPE = 't5'  # 't5', 'mt5', 'byt5'
T5_MODEL_NAME = 't5-base'
T5_ARGS = T5Args(
    model_type=T5_MODEL_TYPE,
    # overwrite_output_dir=True,

    train_batch_size=3  # CUDA out of memory if greater ;/
)
#####################################################################
BERT_MODEL_TYPE = 'roberta'  # bert, roberta, xlm, ...
BERT_MODEL_NAME = 'roberta-base'
BERT_ARGS = ClassificationArgs(
    model_type=BERT_MODEL_TYPE,
    # overwrite_output_dir=True

)
