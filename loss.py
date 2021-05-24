import torch
import numpy as np
from torch import nn


class RelationalNeigborConstrastLoss(nn.Module):
    def __init__(self):
        super(RelationalNeigborConstrastLoss, self).__init__()

    def forward(self, typeA_idx, typeB_idx, typeA_encoding, typeB_encoding, typeA_adj, typeB_adj):

        typeA_anchor, typeA_positive = self.negative_sampler(typeA_idx, typeB_idx, typeB_adj)
        typeB_anchor, typeB_positive = self.negative_sampler(typeB_idx, typeA_idx, typeA_adj)

        typeA_loss = self.n_pairs_loss(typeA_encoding, typeB_encoding, typeA_anchor, typeA_positive)
        typeB_loss = self.n_pairs_loss(typeB_encoding, typeA_encoding, typeB_anchor, typeB_positive)

        return typeA_loss + typeB_loss

    @staticmethod
    def negative_sampler(typeA_idx, typeB_idx, typeB_adj):
        # typeA_idx, typeB_idx: index of samples in the original relational matrix

        typeA_idx = typeA_idx.cpu().numpy()
        typeB_idx = typeB_idx.cpu().numpy()
        typeB_adj = typeB_adj.cpu().numpy()

        typeA_anchor = []
        typeA_positive = []


        num_pairs = len(typeA_idx)

        for i in range(num_pairs):
            typeA_anchor.append(np.array(typeA_idx[i]))

            typeB_neighbor_idx = list(np.where(typeB_adj[:, typeB_idx[i]] != 0)[0])

            typeA_positive.append([typeB_idx[i]] + typeB_neighbor_idx)

        typeA_anchor = np.array(typeA_anchor)
        typeA_positive = np.array(typeA_positive)

        return torch.LongTensor(typeA_anchor), typeA_positive

    @staticmethod
    def n_pairs_loss(typeA_encoding, typeB_econding, typeA_anchor, typeA_poistive):
        loss = 0.0
        count = 1.0
        for i in range(typeA_anchor.shape[0]):
            n_positive_ = torch.LongTensor(typeA_poistive[i])
            n_anchor_ = typeA_anchor[i]

            if typeA_encoding.is_cuda:
                n_positive_.cuda()
                n_anchor_.cuda()

            anchors = typeA_encoding[n_anchor_, :]
            # positives = embeddings[n_positive_].mean(dim=0, keepdims=True)
            query_positive = typeB_econding[n_positive_[0], :]

            # key_positive = typeB_econding[n_positive_[1:], :].detach()
            key_positive = typeB_econding[n_positive_[1:], :]
            if key_positive.nelement() == 0:
                continue

            # positives = query_positive.view(1, -1) + key_positive.mean(dim=0, keepdims=True)
            weight = len(key_positive.data)*torch.softmax(torch.matmul(query_positive.data, key_positive.data.T), dim=0)

            # prox = torch.matmul(query_positive.data, key_positive.data.T)
            # prox = torch.matmul(query_positive, key_positive.T)
            pos_part = torch.matmul(anchors, key_positive.transpose(0, 1))
            # x = torch.sum(weight*torch.abs(pos_part - 1))
            # x = torch.sum(torch.exp(x), 1)
            x = torch.sum(weight*(pos_part - 1)**2)
            # x = torch.sum((pos_part - prox) ** 2)
            loss += x
            count += 1
        loss = loss / count

        return loss


class NPairLossCenter(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """
    def __init__(self, l2_reg=0.02):
        super(NPairLossCenter, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_anchors, n_positives = self.get_n_pairs(target, embeddings.size(0))

        if len(n_anchors) == 0:
            losses = 0
        else:
            losses = self.n_pair_loss(embeddings, n_anchors, n_positives) \
                     + self.l2_reg * self.l2_loss(embeddings)

        return losses

    @staticmethod
    def get_n_pairs(labels, sample_num):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.view(sample_num, -1).cpu().data.numpy()
        n_anchor = []
        n_positive = []

        for i in range(sample_num):
            positive_mask = (labels[i, :] != 0.0)

            positive_indices = np.where(positive_mask)[0]
            if len(positive_indices) < 1:
                continue
            # positive = np.random.choice(positive_indices, len(positive_indices), replace=False)
            anchor = np.array([i])
            n_anchor.append(anchor)
            n_positive.append(positive_indices)


        n_anchor = np.array(n_anchor)
        n_positive = np.array(n_positive)

        return torch.LongTensor(n_anchor), n_positive

    @staticmethod
    def n_pair_loss(embeddings, n_anchors, n_positives,  target=None):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        # anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        # positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
        loss = 0
        for i in range(n_anchors.size(0)):
            n_positive_ = torch.LongTensor(n_positives[i])
            n_anchor_ = n_anchors[i]
            if embeddings.is_cuda:
                n_anchor_.cuda()
            anchors = embeddings[n_anchor_]

            if target is not None:
                coeff = target[n_anchor_, n_positive_]
                coeff = coeff.view(1, -1) / coeff.sum()
                positives = torch.matmul(coeff, embeddings[n_positive_])
            else:
                positives = embeddings[n_positive_]
                # pos_part = torch.sum(torch.exp(-torch.matmul(anchors, positives.transpose(0, 1))))
            pos_part = torch.matmul(anchors, positives.transpose(0, 1))
            x = torch.sum(torch.abs(pos_part - 1))
            loss += x
            # loss += pos_part
            # x = torch.matmul(anchors, (negatives - positives).transpose(0, 1))
            # x = torch.exp(x)
        
        loss = loss / n_anchors.size(0)

        return loss


    @staticmethod
    def l2_loss(anchors):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2) / anchors.shape[0]


def dda_loss(y_predict, y_true):
    # ce_loss = nn.CrossEntropyLoss()
    # return ce_loss(y_predict, y_true)
    # mse_loss = nn.MSELoss(reduction="sum")
    # return mse_loss(y_predict, y_true)
    binary_loss = nn.BCELoss(reduction="sum")
    return binary_loss(y_predict, y_true)


def balance_dda_loss(y_predict, y_true):

    pos_num = torch.sum(y_true)
    neg_num = y_true.size(0) - pos_num

    pos_weight = neg_num / (pos_num + neg_num)
    neg_weight = pos_num / (pos_num + neg_num)

    mse_loss = nn.MSELoss(reduction="sum")
    pos_part = mse_loss(y_predict[y_true == 1], y_true[y_true == 1])
    neg_part = mse_loss(y_predict[y_true == 0], y_true[y_true == 0])
    # return pos_part+neg_part

    # binary_loss = nn.BCELoss(reduction="sum")
    # pos_part = binary_loss(y_predict[y_true == 1], y_true[y_true == 1])
    # neg_part = binary_loss(y_predict[y_true == 0], y_true[y_true == 0])

    return pos_weight*pos_part + neg_weight*neg_part

