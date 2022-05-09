import os
from simpletransformers.t5 import T5Args

DATA_DIR = 'data'
TRAIN_DATA = os.path.join(DATA_DIR, 'train_downloaded.csv')
TEST_DATA = os.path.join(DATA_DIR, 'test_downloaded.csv')

T5_OUTPUT = 't5-output'
BERT_OUTPUT = 'bert-output'

T5_MODEL_TYPE = 't5'  # 't5', 'mt5', 'byt5'
T5_MODEL_NAME = 't5-base'

INIT_TRAIN_SIZE = 300
MAX_TRAIN_SIZE = 2700
STEP = 400

t5_args = T5Args(
    model_type=T5_MODEL_TYPE,
    overwrite_output_dir=True,

    num_train_epochs=1,
    train_batch_size=3  # CUDA out of memory if greater ;/
)
