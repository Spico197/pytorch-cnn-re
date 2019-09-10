import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPPER_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# ----------------- FILES -------------------------
TRAIN_FILE = os.path.join(UPPER_DIR, 'data/train.txt')
DEV_FILE = os.path.join(UPPER_DIR, 'data/dev.txt')
TEST_FILE = os.path.join(UPPER_DIR, 'data/test.txt')
WORD2VEC_PATH = os.path.join(UPPER_DIR, 'middle/word.vec')

# ----------- DATA PREPARATION --------------------
WORD_EMBEDDING_DIM = 50
POS_EMBEDDING_DIM = 10
WORD2VEC_WINDOW = 5
MAX_SENT_LEN = 150

# ----------------- TRAIN -------------------------
EPOCH = 10
EARLY_STOP_EPOCH = 5
BATCH_SIZE = 64
SEED = 2019
HIDDEN_SIZE = 220
KERNEL_SIZE = 3
PRINT_PER_STEP = 10
LEARNING_RATE = 1e-3
BEST_MODEL_SAVE_PATH = os.path.join(UPPER_DIR, 'result/best_model.pth')
