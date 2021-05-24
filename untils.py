import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def chck_dir(tdir):
    if not os.path.exists(tdir):
        os.makedirs(tdir)


def compute_adj(X, topk):
    X = X.numpy()
    sample_num = X.shape[0]
    t_adj_mat = np.zeros((sample_num, sample_num))
    for i in range(sample_num):
        dist = np.diag(np.dot(X[i, :] - X, (X[i, :] - X).T))
        # dist = -X[i, :]
        # dist[i] = 0
        ind = np.argsort(dist)
        t_adj_mat[i, ind[:topk]] = 1

    adj_mat_bool = ((t_adj_mat + t_adj_mat.T) / 2) > 0.5

    sym_adj_mat = np.zeros((sample_num, sample_num))
    sym_adj_mat[adj_mat_bool] = 1.0

    return sym_adj_mat - np.diag(np.diag(sym_adj_mat))


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind // array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


def get_k_fold(k, i, X, y):
    # 返回第i折交叉验证是所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None

    for j in range(k):

        idx = slice(j*fold_size, (j+1) * fold_size)

        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid


def flatten_data(disease_data, drug_data, get_idx=False):
    disease_num, disease_dim = disease_data.shape[0], disease_data.shape[1]
    drug_num, drug_dim = drug_data.shape[0], drug_data.shape[1]
    disease_drug_assoc = torch.zeros((disease_num, drug_num))

    idx = torch.where(disease_drug_assoc == 0)

    if get_idx:
        return idx
    else:
        disease_drug_data = torch.cat((disease_data[idx[0]], drug_data[idx[1]]), dim=1)
        return disease_drug_data


def data_iter_obsolte(pos_X_train_idx, unkwn_pairs_idx, batch_size, neg_pos_ratio):
    np.random.shuffle(unkwn_pairs_idx)
    pos_num = len(pos_X_train_idx)
    neg_num = np.minimum(neg_pos_ratio*pos_num, len(unkwn_pairs_idx))

    selected_unkwn_paris_idx = unkwn_pairs_idx[0:neg_num]

    y = torch.cat((torch.ones(pos_num, ), torch.zeros(neg_num, )), dim=0)
    #
    # X = torch.cat((disease_drug_data[pos_X_train_idx, :],
    #                disease_drug_data[selected_unkwn_paris_idx, :]), dim=0)
    X = torch.cat((torch.tensor(pos_X_train_idx), torch.tensor(selected_unkwn_paris_idx)), dim=0)
    dataset = TensorDataset(X, y)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    return train_iter


def data_loader(pos_X_train_idx, unkwn_pairs_idx, batch_size, neg_pos_ratio, neg_sample_weight):
    # np.random.seed(123)
    # np.random.shuffle(unkwn_pairs_idx)
    pos_num = len(pos_X_train_idx)
    neg_num = np.minimum(neg_pos_ratio * pos_num, len(unkwn_pairs_idx))

    # selected_unkwn_paris_idx = unkwn_pairs_idx[0:neg_num]
    select_unkwn_pairs_idx = np.random.choice(unkwn_pairs_idx, neg_num, replace=False, p=neg_sample_weight)

    pos_train_iter = data_iter(pos_X_train_idx, batch_size, pos=True)
    neg_train_iter = data_iter(select_unkwn_pairs_idx, batch_size*neg_pos_ratio, pos=False)

    if len(pos_train_iter) != len(neg_train_iter):
        assert "pos-neg missmathced!"

    return zip(pos_train_iter, neg_train_iter), select_unkwn_pairs_idx


def data_iter(train_idx, batch_size, pos=True):
    if pos:
        y = torch.ones((len(train_idx), ))
    else:
        y = torch.zeros(len(train_idx, ))
    dataset = TensorDataset(torch.tensor(train_idx), y)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    return train_iter


def loss_in_epoch(train_ce_ls, train_roc_ls, train_pr_ls, test_ce_ls, test_roc_ls, test_pr_ls,
                  title_, fout1, fout2, num_epochs, interval):
    # epoch_num = len(train_ce_ls)
    fig1, ax1 = plt.subplots()
    ax1.plot(range(interval, num_epochs+interval, interval), train_ce_ls, "r--", label="Train Loss")
    ax1.plot(range(interval, num_epochs+interval, interval), test_ce_ls, "b", label="Test Loss")
    # ax1.plot(range(1, epoch_num+1), test_acc_ls, "b:", label="Test ACC")
    ax1.set(xlabel="Epoch", ylabel="Loss", title=title_)
    lg1 = ax1.legend(loc='best')
    fig1.savefig(fout1)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(interval, num_epochs+interval, interval), train_roc_ls, "r", label="Train ROC")
    ax2.plot(range(interval, num_epochs+interval, interval), test_roc_ls, "b", label="Test ROC")
    ax2.plot(range(interval, num_epochs+interval, interval), train_pr_ls, "r--", label="Train PR")
    ax2.plot(range(interval, num_epochs+interval, interval), test_pr_ls, "b--", label="Test PR")

    ax2.set(xlabel="Epoch", ylabel="Metric", title=title_)
    lg2 = ax2.legend(loc='best')
    fig2.savefig(fout2)


def adjust_learning_rate(optimizer, epoch, init_lr):

    if epoch < 100:
        update_lr = init_lr
    elif epoch < 200:
        update_lr = 0.01
    else:
        update_lr = 0.001

    for param_group in optimizer.param_groups:
        param_group["lr"] = update_lr


def cyclial_learning_rate(optimizer, epoch, min_lr, init_max_lr, step, lr_decay):
    k = np.floor(epoch / (2*step))
    max_lr = init_max_lr*lr_decay ** k
    cycle = np.ceil(epoch / (2*step))
    x = np.abs(epoch / step - 2 * cycle + 1)
    # if epoch > 1500:
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, 1-x)
    #    lr = init_max_lr / 9
    # elif epoch > 1000:
    #     lr = init_max_lr / 3
    # elif epoch > 500:
    #    lr = init_max_lr / 3
    # else:
    #    lr = init_max_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer


def step_decay_learning_rate(optimizer, epoch, init_lr, step, lr_decay):
    lr = init_lr * (lr_decay ** np.floor(epoch/step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def write_train_record(f_name, train_ls, valid_ls):

    write_lines = ["epoch %d, train loss: %f, test loss: %f\n"
                   % (i+1, train_ls[i], valid_ls[-1]) for i in range(len(train_ls))]

    write_lines = ["="*60+"\n"] + write_lines
    with open(f_name, "a") as fout:
        fout.writelines(write_lines)
