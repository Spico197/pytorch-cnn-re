import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from local_config import *
from model import CNNModel
from evaluate import evaluate
from data_helper import trainset, devset, testset, word_vectors, word2id, rel2id


if __name__ == "__main__":

    train_loader = DataLoader(trainset, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader = DataLoader(devset, shuffle=False, batch_size=BATCH_SIZE)
    test_loader = DataLoader(testset, shuffle=False, batch_size=BATCH_SIZE)

    model = CNNModel(torch.tensor(word_vectors, dtype=torch.float32), rel2id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    criterion.to(DEVICE)

    best_f1_micro = 0.0
    waste_epoch = 0
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            model.train()
            tokens, pos1, pos2, label = data

            tokens = tokens.to(DEVICE)
            pos1 = pos1.to(DEVICE)
            pos2 = pos2.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(tokens, pos1, pos2)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % PRINT_PER_STEP == PRINT_PER_STEP - 1:
                acc, precision, recall, f1_micro, f1_macro = evaluate(model, train_loader, DEVICE)
                print(' [%d, %5d] AVG-Loss: %.4f - TRAIN >>> ACC: %.4f, Precision: %.4f, Recall: %.4f, F1-micro: %.4f, F1-macro: %.4f\r' \
                    % (epoch+1, i+1, running_loss / PRINT_PER_STEP, acc, precision, recall, f1_micro, f1_macro), end='')
                running_loss = 0.0

        acc, precision, recall, f1_micro, f1_macro = evaluate(model, test_loader, DEVICE)
        print('\nTEST >>> ACC: %.4f, Precision: %.4f, Recall: %.4f, F1-micro: %.4f, F1-macro: %.4f\n' \
                % (acc, precision, recall, f1_micro, f1_macro))
        if f1_micro > best_f1_micro:
            print('Best model, storing...\n')
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            best_f1_micro = f1_micro
        else:
            waste_epoch += 1
        if waste_epoch >= EARLY_STOP_EPOCH:
            break

    print('Traning finished. Best f1-micro score: %.4f' % best_f1_micro)
