from math import floor, ceil

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
        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.pos_embedding = nn.Embedding(MAX_SENT_LEN*2, POS_EMBEDDING_DIM)
        pads = (KERNEL_SIZE-1)/2
        if pads - floor(pads) > 0.0:
            padding = (floor(pads), ceil(pads))
        else:
            padding = int(pads)
        self.conv = nn.Conv1d(WORD_EMBEDDING_DIM + 2*POS_EMBEDDING_DIM, FILTER_NUM, KERNEL_SIZE, padding=padding)
        self.max_pool = nn.MaxPool1d(MAX_SENT_LEN, 1)
        self.fc1 = nn.Linear(FILTER_NUM, HIDDEN_SIZE)
        self.fc2 = nn.Linear(FILTER_NUM, len(rel2id))
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def forward(self, tokens, pos1, pos2):
        word_embedding_layer = self.word_embedding(tokens)
        pos1_embedding_layer = self.pos_embedding(pos1)
        pos2_embedding_layer = self.pos_embedding(pos2)
        concat_layer = torch.cat([word_embedding_layer, pos1_embedding_layer, pos2_embedding_layer], dim=-1)
        out = concat_layer.permute(0, 2, 1)
        out = self.conv(out)
        out = F.relu(out)
        out = self.max_pool(out)
        out = out.view(-1, FILTER_NUM)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=-1)
        return out
