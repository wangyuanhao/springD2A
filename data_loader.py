import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_k_fold(k, i, X):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, X_valid = None, None

    sample_num = X.shape[0]
    for j in range(k):

        if j == (k - 1):
            idx = slice(j * fold_size, sample_num)
        else:
            idx = slice(j*fold_size, (j+1) * fold_size)

        X_part= X[idx]
        if j == i:
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)

    return X_train, X_valid


class CVDataLoader():

    def __init__(self, disease_fi, drug_fi, interact_fi):
        typeA_data = pd.read_csv(disease_fi, header=None, index_col=None)
        typeB_data = pd.read_csv(drug_fi, header=None, index_col=None)
        interact = pd.read_csv(interact_fi, header=None, index_col=None)

        self.typeA_data = typeA_data.values
        self.typeB_data = typeB_data.values
        interact = interact.values
        # align with matlab, column first
        # interact = interact.T

        self.interact_idx = np.array([[i, j]
                                 for i in range(typeA_data.shape[0])
                                 for j in range(typeB_data.shape[0])])
        self.interact = interact.reshape(-1, )

        self.unknw_pairs_idx = np.where(self.interact == 0)[0]
        self.unknw_pairs_num = len(self.unknw_pairs_idx)
        self.approved_pairs_idx = np.where(self.interact == 1)[0]
        self.approved_pairs_num = len(self.approved_pairs_idx)

    def flatten_data(self, interact_idx, typeA_data, typeB_data, flatten_idx):
        data_idx = interact_idx[flatten_idx, :]
        typeA_data_idx = data_idx[:, 0]
        typeB_data_idx = data_idx[:, 1]
        flatten_typeA_data = typeA_data[typeA_data_idx, :]
        flatten_typeB_data = typeB_data[typeB_data_idx, :]

        return flatten_typeA_data, flatten_typeB_data

    def create_dataset(self, neg_pos_ratio):
        interact = self.interact
        approved_pairs_idx = self.approved_pairs_idx

        selected_positive = interact[approved_pairs_idx]

        return selected_positive

    def full_interact(self, pos_idx):

        typeA_data = self.typeA_data
        typeB_data = self.typeB_data

        typeA_num = typeA_data.shape[0]
        typeB_num = typeB_data.shape[0]

        interact = np.zeros((typeA_num*typeB_num, ))
        interact[pos_idx] = 1

        full_interact_ = interact.reshape((typeA_num, typeB_num))

        return full_interact_

    def cross_validation_iter(self, kfold):

        approved_pairs_idx = self.approved_pairs_idx
        approved_pairs_idx = shuffle(approved_pairs_idx, random_state=100)

        unkwn_paris_idx = self.unknw_pairs_idx

        for k in range(kfold):

            pos_X_train_idx, pos_X_test_idx = get_k_fold(kfold, k, approved_pairs_idx)

            Y_train_interact = self.full_interact(pos_X_train_idx)
            Y_test_interact = self.full_interact(pos_X_test_idx)

            yield pos_X_train_idx, pos_X_test_idx, unkwn_paris_idx, Y_train_interact, Y_test_interact

