import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from local_config import *


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class CNNModel(nn.Module):
    def __init__(self, word_vectors, rel2id):
        super(CNNModel, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_vectors)
        self.pos_embedding = nn.Embedding(MAX_SENT_LEN*2, POS_EMBEDDING_DIM)
        self.conv = nn.Conv1d(WORD_EMBEDDING_DIM + 2*POS_EMBEDDING_DIM, HIDDEN_SIZE, KERNEL_SIZE)
        self.max_pool = nn.MaxPool1d(MAX_SENT_LEN - KERNEL_SIZE + 1, 1)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, len(rel2id))
    
    def forward(self, tokens, pos1, pos2):
        word_embedding_layer = self.word_embedding(tokens)
        pos1_embedding_layer = self.pos_embedding(pos1)
        pos2_embedding_layer = self.pos_embedding(pos2)
        concat_layer = torch.cat([word_embedding_layer, pos1_embedding_layer, pos2_embedding_layer], dim=-1)
        y = concat_layer.permute(0, 2, 1)
        y = self.conv(y)
        y = F.relu(y)
        y = self.max_pool(y)
        y = y.view(-1, HIDDEN_SIZE)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        return y
