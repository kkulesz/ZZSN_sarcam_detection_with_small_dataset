import pandas as pd
from simpletransformers.t5 import T5Args
import warnings

from models.t5 import T5Wrapper
from models.bert import BertWrapper
from trainer import OurTrainer
import consts

# ugly, but simpletransfomers.T5 throws some stupid
# deprecation warnings if everything is done the way
# the official tutorial says: https://simpletransformers.ai/docs/t5-model/
warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_t5(number_of_rows: int) -> T5Wrapper:
    args = T5Args(
        model_type=consts.T5_MODEL_TYPE,
        output_dir=f"{consts.T5_OUTPUT}-{number_of_rows}",
        overwrite_output_dir=True,

        num_train_epochs=1,
        train_batch_size=3,  # CUDA out of memory if greater ;/
    )
    return T5Wrapper(load_pretrained=False, args=args)


def prepare_bert() -> BertWrapper:
    pass


if __name__ == '__main__':
    nrows = 10
    raw_train_data = pd.read_csv(consts.TRAIN_DATA, nrows=nrows)

    model = prepare_t5(nrows)
    trainer = OurTrainer(model)
    trainer.train(raw_train_data)
