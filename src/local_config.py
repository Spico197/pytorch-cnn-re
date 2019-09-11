import os
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPPER_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# ----------------- FILES -------------------------
TRAIN_FILE = os.path.join(UPPER_DIR, 'data/merge_train.txt')
DEV_FILE = os.path.join(UPPER_DIR, 'data/dev.txt')
TEST_FILE = os.path.join(UPPER_DIR, 'data/test.txt')
WORD2VEC_PATH = os.path.join(UPPER_DIR, 'middle/word.vec')

# ----------- DATA PREPARATION --------------------
PRETRAINED_WORD_VECTOR_PATH = 'E:\\Data\\Embeddings\\glove.6B.50d.txt'
WORD_EMBEDDING_DIM = 50
POS_EMBEDDING_DIM = 50
WORD2VEC_WINDOW = 5
MAX_SENT_LEN = 150

# ----------------- TRAIN -------------------------
GPU_DEVICE_NUM = 0
DEVICE = torch.device("cuda:{}".format(GPU_DEVICE_NUM) if torch.cuda.is_available() else "cpu")

EPOCH = 100
EARLY_STOP_EPOCH = -1
BATCH_SIZE = 64
SEED = 2019

FILTER_NUM = 220
HIDDEN_SIZE = 100
KERNEL_SIZE = 3
PRINT_PER_STEP = 10
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
BEST_MODEL_SAVE_PATH = os.path.join(UPPER_DIR, 'result/best_model.pth')
TEST_PRC_FIGURE_SAVE_PATH = os.path.join(UPPER_DIR, 'result/test_prc.png')
