import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_helper import trainset, devset, testset, word_vectors, word2id, rel2id
from local_config import *


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class CNNModel(nn.Module):
    def __init__(self, word_vectors):
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


def evaluate(model, test_loader):
    correct = total = 0
    for data in test_loader:
        tokens, pos1, pos2, labels = data
        outputs = model(tokens, pos1, pos2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print(100*float(correct.item())/total)


train_loader = DataLoader(trainset, shuffle=True, batch_size=BATCH_SIZE)
dev_loader = DataLoader(devset, shuffle=False, batch_size=BATCH_SIZE)
test_loader = DataLoader(testset, shuffle=False, batch_size=BATCH_SIZE)

model = CNNModel(torch.tensor(word_vectors, dtype=torch.float32))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        tokens, pos1, pos2, label = data
        optimizer.zero_grad()
        outputs = model(tokens, pos1, pos2)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % PRINT_PER_STEP == PRINT_PER_STEP - 1:
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / PRINT_PER_STEP))
            running_loss = 0.0

    with torch.no_grad():
        evaluate(model, dev_loader)

print('testing...')
with torch.no_grad():
        evaluate(model, test_loader)
print('chance guessing rate is: %.2f' % (100.0/len(rel2id)))
