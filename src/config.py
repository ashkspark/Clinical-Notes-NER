from transformers import BertTokenizer

CONCEPT_FILES_PATH = "../input/concepts"
DOCUMENT_FILES_PATH = "../input/docs"

TRAIN_FILE = "../input/train/train.csv"
MODEL_FILE = "../models/model.bin"
META_DATA_FILE = "../models/meta.bin"

MAX_LEN = 128
TOKENIZER = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower=True
)

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 3