from transformers import BertTokenizer

CONCEPT_FILES_PATH_BI = "../input/beth_israel/concepts"
DOCUMENT_FILES_PATH_BI = "../input/beth_israel/docs"
CONCEPT_FILES_PATH_PA = "../input/partners/concepts"
DOCUMENT_FILES_PATH_PA = "../input/partners/docs"

TRAIN_FILE = "../input/train/train.csv"
MODEL_FILE = "../models/model.bin"
META_DATA_FILE = "../models/meta.bin"

MAX_LEN = 256
MAX_WAITING = 3
TOKENIZER = BertTokenizer.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    do_lower=True
)

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 3

TRAIN_CHUNK_SIZE = 50
TEST_CHUNK_SIZE = 50