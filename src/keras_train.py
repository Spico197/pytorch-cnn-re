import numpy as np
import keras
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, \
                            Activation, Flatten, concatenate)
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from torch.utils.data import DataLoader

from local_config import *
from data_helper import word_vectors, trainset, devset, testset, rel2id


word_vectors = np.array(word_vectors)

words_input = Input(shape=(MAX_SENT_LEN, ), dtype='int32', name='words_input')
words = Embedding(word_vectors.shape[0], word_vectors.shape[1],
                    weights=[word_vectors], trainable=True)(words_input)

distance1_input = Input(shape=(MAX_SENT_LEN, ), dtype='int32', name='distance1_input')
distance1 = Embedding(MAX_SENT_LEN*2, POS_EMBEDDING_DIM)(distance1_input)

distance2_input = Input(shape=(MAX_SENT_LEN, ), dtype='int32', name='distance2_input')
distance2 = Embedding(MAX_SENT_LEN*2, POS_EMBEDDING_DIM)(distance2_input)

output = concatenate([words, distance1, distance2])
output = Convolution1D(filters=FILTER_NUM,
                        kernel_size=KERNEL_SIZE,
                        padding='same',
                        activation='relu',
                        strides=1)(output)

output = GlobalMaxPooling1D()(output)

output = Dropout(0.5)(output)
output = Dense(len(rel2id), activation='softmax')(output)

model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
model.compile(loss='categorical_crossentropy', 
                optimizer='Adam',
                metrics=['accuracy'])
model.summary()

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

def predict_classes(prediction):
    return prediction.argmax(axis=-1)

train_loader = DataLoader(trainset)
test_loader = DataLoader(testset)

def get_data(dataloader):
    token_test = []
    pos1_test = []
    pos2_test = []
    labels_test = []
    for tokens, pos1, pos2, labels in dataloader:
        tokens = tokens.numpy()
        pos1 = pos1.numpy()
        pos2 = pos2.numpy()
        labels = labels.numpy()
        if len(token_test) == 0:
            token_test = tokens
            pos1_test = pos1
            pos2_test = pos2
            labels_test = labels
        else:
            token_test = np.concatenate([token_test, tokens], axis=0)
            pos1_test = np.concatenate([pos1_test, pos1], axis=0)
            pos2_test = np.concatenate([pos2_test, pos2], axis=0)
            labels_test = np.concatenate([labels_test, labels], axis=0)
    return token_test, pos1_test, pos2_test, labels_test

print('get train data')
train_tokens, train_pos1, train_pos2, train_labels = get_data(train_loader)
print('get test data')
test_tokens, test_pos1, test_pos2, test_labels = get_data(test_loader)

print('training')
for epoch in range(EPOCH):
    #TODO: fix data organization bugs
    raise NotImplementedError
    model.fit([train_tokens, train_pos1, train_pos2], train_labels.T, batch_size=BATCH_SIZE, verbose=True, epochs=1)   
    pred_test = predict_classes(model.predict([test_tokens, test_pos1, test_pos2], verbose=False))
    print(pred_test)
    assert 0
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(test_labels)

    acc =  np.sum(pred_test == test_labels) / float(len(test_labels))
    max_acc = max(max_acc, acc)
    print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

    f1Sum = 0
    f1Count = 0
    for targetLabel in range(1, max(yTest)):        
        prec = getPrecision(pred_test, yTest, targetLabel)
        recall = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
        f1Sum += f1
        f1Count +=1    
        
        
    macroF1 = f1Sum / float(f1Count)    
    max_f1 = max(max_f1, macroF1)
    print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))

