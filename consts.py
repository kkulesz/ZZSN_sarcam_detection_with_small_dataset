import os
from simpletransformers.t5 import T5Args
from simpletransformers.classification import ClassificationArgs

T5 = 't5'
BERT = 'bert'
CURRENT_VARIANT = BERT
TASK_DETECT = 'binary sarcasm clasification'
TASK_CLASIFICATION = 'sarcas type clasification'
TASK_TEST = 'learning test'
CURRENT_TASK = TASK_CLASIFICATION
####################################################################
DATA_DIR = 'data'
TRAIN_DATA = os.path.join(DATA_DIR, 'train_preprocessed.csv')
TEST_DATA = os.path.join(DATA_DIR, 'test_preprocessed.csv')

TRAIN_SARCASM_DATA = os.path.join(DATA_DIR, 'train_sarcasm_preprocessed.csv')
TEST_SARCASM_DATA = os.path.join(DATA_DIR, 'test_sarcasm_preprocessed.csv')

TRAIN_TESTING_DATA = os.path.join(DATA_DIR, 'learning_test.csv')
TEST_TESTING_DATA = os.path.join(DATA_DIR, 'learning_train.csv')

DATA_DIRS = {
    'TASK_DETECT_TRAIN': TRAIN_DATA, 'TASK_DETECT_TEST': TEST_DATA,
    'TASK_CLASIFICATION_TRAIN': TRAIN_SARCASM_DATA, 'TASK_CLASIFICATION_TEST': TEST_SARCASM_DATA,
    'TASK_TEST_TRAIN': TRAIN_TESTING_DATA, 'TASK_TEST_DATA': TEST_TESTING_DATA,
}

T5_OUTPUT = 't5-output'
BERT_OUTPUT = 'bert-output'

INIT_TRAIN_SIZE = 500
MAX_TRAIN_SIZE = 500
STEP = 500
SAVE_EVERY_N_EPOCHS=10
#####################################################################
T5_MODEL_TYPE = 't5'  # 't5', 'mt5', 'byt5'
T5_MODEL_NAME = 't5-base'
T5_ARGS = T5Args(
    model_type=T5_MODEL_TYPE,
    # overwrite_output_dir=True,

    train_batch_size=3,  # CUDA out of memory if greater ;/
    num_train_epochs=100,
    save_model_every_epoch=False,
    save_steps=INIT_TRAIN_SIZE*SAVE_EVERY_N_EPOCHS
)
#####################################################################
BERT_MODEL_TYPE = 'roberta'  # bert, roberta, xlm, ...
BERT_MODEL_NAME = 'roberta-base'
BERT_ARGS = ClassificationArgs(
    overwrite_output_dir=True,
    num_train_epochs=100,
    logging_steps=1,
    save_model_every_epoch=False,
    save_steps=INIT_TRAIN_SIZE*SAVE_EVERY_N_EPOCHS
)
