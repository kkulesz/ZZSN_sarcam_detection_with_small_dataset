import os

DATA_DIR = 'data'
TRAIN_DATA = os.path.join(DATA_DIR, 'train_downloaded.csv')
TEST_DATA = os.path.join(DATA_DIR, 'test_downloaded.csv')

T5_OUTPUT = 't5-output'
BERT_OUTPUT = 'bert-output'

T5_MODEL_TYPE = 't5'  # 't5', 'mt5', 'byt5'
