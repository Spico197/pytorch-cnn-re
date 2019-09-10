import re
import os
import multiprocessing
import pickle

import numpy as np
import torch
from nltk import word_tokenize
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

from local_config import *


os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class SemEvalUtils(object):
    def __init__(self, files, **kwargs):
        self.files = files
        self.word_vectors = list()
        self.word2id = dict()
        self.model = None
        self.rel2id = dict()
        self.id2rel = dict()
        self.sents = list()
        self.rels = list()
        self._init_construction()

    def _load_data(self, filepath):
        data = []
        tmp_d = dict()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    data.append(tmp_d)
                    tmp_d = dict()
                else:
                    if line.startswith('Comment:'):
                        tmp_d['comment'] = line[8:].strip()
                    elif '</e1>' in line:
                        tmp_d['id'] = line.split('\t')[0]
                        tmp_d['sentence'] = line.split('\t')[-1][1:-1]
                        tmp_d['head'] = re.search(r'<e1>(.*?)</e1>', line).group(1)
                        tmp_d['tail'] = re.search(r'<e2>(.*?)</e2>', line).group(1)
                    elif '(e1' in line or '(e2' in line or 'Other' in line:
                        tmp_d['relation'] = line

        cnt = 1
        res_data = []
        for d in data:
            head = d['head']
            tail = d['tail']

            new_sent = re.sub(r'<e1>.*?</e1>', ' {} '.format(d['head']), d['sentence'])
            new_sent = re.sub(r'<e2>.*?</e2>', ' {} '.format(d['tail']), new_sent)
            new_sent = ' '.join(word_tokenize(new_sent))

            head_pos = [[new_sent.split().index(head.split()[0]), \
                                new_sent.split().index(head.split()[0]) + len(head.split())]]
            tail_pos = [[new_sent.split().index(tail.split()[0]), \
                            new_sent.split().index(tail.split()[0]) + len(tail.split())]]
            one_data = {
                "id": cnt,
                "ori_id": d['id'],
                "head_pos": head_pos,
                "tail_pos": tail_pos,
                "token": new_sent.lower().split(),
                "sentence": new_sent.lower(),
                "relation": d['relation'],
                "head": head,
                "tail": tail
            }
            res_data.append(one_data)
            cnt += 1
        return res_data

    def _init_construction(self):
        self.vocab = set()
        for file in self.files:
            data = self._load_data(file)
            for d in data:
                self.sents.append(d['sentence'])
                self.rels.append(d['relation'])
                for word in d['token']:
                    self.vocab.add(word)
        relations = set(self.rels)
        self.rel2id = {rel:ind for ind, rel in enumerate(relations)}
        self.id2rel = {ind:rel for ind, rel in enumerate(relations)}
        self.vocab = list(self.vocab)
        self.vocab.append('oov')
        self.vocab.append('unk')
        self.word2id = {word:i for i, word in enumerate(self.vocab)}

    def train_word_vec(self, **kwargs):
        dim = kwargs.get('dim', WORD_EMBEDDING_DIM)
        window = kwargs.get('window', WORD2VEC_WINDOW)
        workers = kwargs.get('workers', multiprocessing.cpu_count())

        self.model = Word2Vec(self.sents, size=dim, window=window, workers=workers)
        self.vocab = list(set(self.model.wv.vocab.keys()))
        self.vocab.append('oov')
        self.vocab.append('unk')
        self.word2id = {word:i for i, word in enumerate(self.vocab)}
        self.word_vectors = np.vstack([self.model.wv[x] for x in self.vocab[:-2]])
        self.word_vectors = np.vstack([self.word_vectors, np.random.uniform(size=(2, dim))])
        return self.word2id, self.word_vectors, self.rel2id, self.id2rel

    def load_pretrained_word_vec(self, path, dim=WORD_EMBEDDING_DIM):
        vocab = set(self.vocab)
        if path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as fin:
                word2vec = dict()
                words = set()
                for line in fin:
                    line = line.strip()
                    strings = line.split()
                    if len(strings) < 1 + WORD_EMBEDDING_DIM:
                        continue
                    else:
                        word = strings[0]
                        if word in vocab:
                            words.add(word)
                            word2vec[word] = [float(x) for x in strings[1: 1 + WORD_EMBEDDING_DIM]]
                        else:
                            continue
                #TODO: fix
                self.vocab = list(words)
                self.vocab.append('oov')
                self.vocab.append('unk')
                self.word2id = {word: ind for ind, word in enumerate(self.vocab)}
                self.word_vectors = np.array([word2vec[word] for word in self.vocab[:-2]])
                self.word_vectors = np.vstack([self.word_vectors, np.random.uniform(size=(2, dim))])
            return self.word2id, self.word_vectors, self.rel2id, self.id2rel
        else:
            raise NotImplementedError

    def save(self, path):
        with open(path, 'wb') as f:
            obj = {
                'word2id': self.word2id, 
                'word_vectors': self.word_vectors,
                'rel2id': self.rel2id,
                'id2rel': self.id2rel,
            }
            pickle.dump(obj, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word2id = data.get('word2id', dict())
        self.word_vectors = data.get('word_vectors', list())
        self.rel2id = data.get('rel2id', dict())
        self.id2rel = data.get('id2rel', dict())
        return self.word2id, self.word_vectors, self.rel2id, self.id2rel


class SemEvalDataSet(Dataset):
    def __init__(self, path, word2id, rel2id):
        self.rel2id = rel2id
        self.word2id = word2id
        self._data = self._load_data(path)
        self.tokens = list()
        self.pos1 = list()
        self.pos2 = list()
        self.labels = list()

        for d in self._data:
            self.tokens.append(d['token'])
            self.labels.append(d['relation'])
            self.pos1.append(d['head_pos_list'])
            self.pos2.append(d['tail_pos_list'])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return torch.tensor(self.tokens[index], dtype=torch.int64), \
                torch.tensor(self.pos1[index], dtype=torch.int64), \
                torch.tensor(self.pos2[index], dtype=torch.int64), \
                torch.tensor(self.labels[index], dtype=torch.int64)

    def _load_data(self, filepath):
        data = []
        tmp_d = dict()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    data.append(tmp_d)
                    tmp_d = dict()
                else:
                    if line.startswith('Comment:'):
                        tmp_d['comment'] = line[8:].strip()
                    elif '</e1>' in line:
                        tmp_d['id'] = line.split('\t')[0]
                        tmp_d['sentence'] = line.split('\t')[-1][1:-1]
                        tmp_d['head'] = re.search(r'<e1>(.*?)</e1>', line).group(1)
                        tmp_d['tail'] = re.search(r'<e2>(.*?)</e2>', line).group(1)
                    elif '(e1' in line or '(e2' in line or 'Other' in line:
                        tmp_d['relation'] = line

        cnt = 1
        res_data = []
        for d in data:
            head = d['head']
            tail = d['tail']

            new_sent = re.sub(r'<e1>.*?</e1>', ' {} '.format(d['head']), d['sentence'])
            new_sent = re.sub(r'<e2>.*?</e2>', ' {} '.format(d['tail']), new_sent)
            new_sent = ' '.join(word_tokenize(new_sent))

            head_pos = new_sent.split().index(head.split()[0])
            tail_pos = new_sent.split().index(tail.split()[0])
            head_pos_list = [0]*MAX_SENT_LEN
            tail_pos_list = [0]*MAX_SENT_LEN
            ori_tokens = new_sent.lower().split()
            for i in range(min(len(ori_tokens), MAX_SENT_LEN)):
                head_pos_list[i] = i - head_pos + MAX_SENT_LEN
            for i in range(min(len(ori_tokens), MAX_SENT_LEN)):
                tail_pos_list[i] = i - tail_pos + MAX_SENT_LEN
            tokens = [self.word2id['unk']]*MAX_SENT_LEN
            for i in range(min(len(ori_tokens), MAX_SENT_LEN)):
                tokens[i] = self.word2id.get(ori_tokens[i], self.word2id['oov'])

            one_data = {
                "id": str(cnt),
                "ori_id": d['id'],
                "head_pos_list": head_pos_list,
                "tail_pos_list": tail_pos_list,
                "token": tokens,
                "relation": self.rel2id[d['relation']],
            }
            res_data.append(one_data)
            cnt += 1
        return res_data


utils = SemEvalUtils([TRAIN_FILE, DEV_FILE, TEST_FILE])
if os.path.exists(WORD2VEC_PATH):
    word2id, word_vectors, rel2id, id2rel = utils.load(WORD2VEC_PATH)
else:
    if os.path.exists(PRETRAINED_WORD_VECTOR_PATH):
        word2id, word_vectors, rel2id, id2rel = utils.load_pretrained_word_vec(PRETRAINED_WORD_VECTOR_PATH, dim=WORD_EMBEDDING_DIM)
    else:
        word2id, word_vectors, rel2id, id2rel = utils.train_word_vec(dim=WORD_EMBEDDING_DIM, window=WORD2VEC_WINDOW)
    utils.save(WORD2VEC_PATH)

trainset = SemEvalDataSet(TRAIN_FILE, word2id, rel2id)
devset = SemEvalDataSet(DEV_FILE, word2id, rel2id)
testset = SemEvalDataSet(TEST_FILE, word2id, rel2id)


if __name__ == "__main__":
    utils = SemEvalUtils([TRAIN_FILE, DEV_FILE, TEST_FILE])
    if os.path.exists(WORD2VEC_PATH):
        word2id, word_vectors, rel2id, id2rel = utils.load(WORD2VEC_PATH)
    else:
        if os.path.exists(PRETRAINED_WORD_VECTOR_PATH):
            word2id, word_vectors, rel2id, id2rel = utils.load_pretrained_word_vec(PRETRAINED_WORD_VECTOR_PATH, dim=WORD_EMBEDDING_DIM)
        else:
            word2id, word_vectors, rel2id, id2rel = utils.train_word_vec(dim=WORD_EMBEDDING_DIM, window=WORD2VEC_WINDOW)
        utils.save(WORD2VEC_PATH)

    trainset = SemEvalDataSet(TRAIN_FILE, word2id, rel2id)
    devset = SemEvalDataSet(DEV_FILE, word2id, rel2id)
    testset = SemEvalDataSet(TEST_FILE, word2id, rel2id)

    # devdataloader = DataLoader(devset, batch_size=64, shuffle=False, num_workers=2)
    # # import ipdb
    # for data in devdataloader:
    #     tokens, pos1, pos2, label = data
    #     # print(tokensl, pos1.size(), pos2.size(), label.size())
    #     print(len(tokens))
    #     print(tokens[0])
    #     print(tokens[0].size())
    #     assert 0
