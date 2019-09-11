import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc

from model import CNNModel
from evaluate import evaluate
from local_config import *
from data_helper import testset, word_vectors, word2id, rel2id


if __name__ == "__main__":

    test_loader = DataLoader(testset, shuffle=False, batch_size=BATCH_SIZE)

    model = CNNModel(torch.tensor(word_vectors, dtype=torch.float32), rel2id)
    model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))

    acc, precision, recall, f1_micro, f1_macro, prs = evaluate(model, test_loader, torch.device('cpu'), prc=True, report=True)
    ps, rs, ths = prs
    print('TEST >>> ACC: %.4f, Precision: %.4f, Recall: %.4f, F1-micro: %.4f, F1-macro: %.4f\n' \
                % (acc, precision, recall, f1_micro, f1_macro))
    print('TEST >>> AUC: %.4f' % auc(rs, ps))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rs, ps, lw=1.5)
    ax.set_title('Precision vs. Recall')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True)
    fig.savefig(TEST_PRC_FIGURE_SAVE_PATH)
    # plt.show()
