import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from untils import flatten_data

class DDAEvaluation():
    def __init__(self, Y_train, Y_test, predict_score):
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.predict_score = predict_score
        self.disease_num = Y_train.shape[0]

    def test_vs_uknw(self):
        Y_test = self.Y_test
        predict_score = self.predict_score
        Y_train = self.Y_train

        # find drugs with prositive label
        # numbers of diseases asscociated with specific drugs
        disease_num_assco_drug = Y_test.sum(axis=0)

        # remove drugs asscociated with zeros diseases
        Y_test_edit = Y_test[:, disease_num_assco_drug != 0]
        Y_train_edit = Y_train[:, disease_num_assco_drug != 0]
        predict_score = predict_score[:, disease_num_assco_drug != 0]

        valid_drug_num = Y_test_edit.shape[1]

        test_vs_unkw_score = []
        test_vs_unkw_label = []
        for i in range(valid_drug_num):

            tmp_score = predict_score[Y_train_edit[:, i] != 1, i]
            # tmp_score = (tmp_score - tmp_score.min()) / (tmp_score.max() - tmp_score.min())
            tmp_label = Y_test_edit[Y_train_edit[:, i] != 1, i]

            test_vs_unkw_score.append(tmp_score)
            test_vs_unkw_label.append(tmp_label)

        return test_vs_unkw_label, test_vs_unkw_score

    def compute_rank_matrix(self):

        test_vs_unkw_score, test_vs_unkw_label = self.test_vs_uknw()
        valid_drug_num = len(test_vs_unkw_score)

        disease_num = self.disease_num
        rank_matrix = np.zeros((disease_num, valid_drug_num))
        rank_matrix[rank_matrix == 0] = None

        for i in range(valid_drug_num):

            drugscore_test_vs_unkw = test_vs_unkw_score[i]
            druglabel_test_vs_unkw = test_vs_unkw_label[i]

            ord = np.argsort(-drugscore_test_vs_unkw)
            # ord = np.argsort(-druglabel_test_vs_unkw)
            label = druglabel_test_vs_unkw[ord]

            rank_matrix[0:len(label), i] = label

        return rank_matrix

    def auroc_pr(self, test_vs_unkw_label, test_vs_unkw_score):
        valid_drug_num = len(test_vs_unkw_score)
        auroc, aupr = np.zeros((valid_drug_num, )), np.zeros((valid_drug_num, ))

        for i in range(valid_drug_num):
            auroc[i] = roc_auc_score(test_vs_unkw_label[i], test_vs_unkw_score[i])
            aupr[i] = average_precision_score(test_vs_unkw_label[i], test_vs_unkw_score[i], pos_label=1)

        return auroc.mean(), aupr.mean()

    def mAP_rec(self, N):

        Y_test = self.Y_test
        Y_train = self.Y_train
        predict_score = self.predict_score

        # find drugs with prositive label
        # numbers of diseases asscociated with specific drugs
        disease_num_assco_drug = Y_test.sum(axis=0)

        # remove drugs asscociated with zeros diseases
        Y_test_edit = Y_test[:, disease_num_assco_drug!=0]
        Y_train_edit = Y_train[:, disease_num_assco_drug!=0]
        predict_score = predict_score[:, disease_num_assco_drug!=0]

        valid_drug_num = Y_test_edit.shape[1]

        map = []
        recall = []
        for i in range(valid_drug_num):
            tmp_score = predict_score[Y_train_edit[:, i] != 1, i]
            tmp_label = Y_test_edit[Y_train_edit[:, i] != 1, i]

            sort_idx = np.argsort(-tmp_score)
            topN_sort_idx = sort_idx[0:N]
            topN_label = tmp_label[topN_sort_idx]

            testIdx = np.nonzero(topN_label == 1)[0]

            if len(testIdx) == 0:
                map.append(0.0)
                recall.append(0.0)
            else:
                precision = np.arange(1, len(testIdx) + 1) / (testIdx + 1)
                map_ = np.sum(precision) / np.sum(tmp_label == 1)
                map.append(map_)
                recall_ = len(testIdx) / np.sum(tmp_label == 1)
                recall.append(recall_)
        return np.mean(map), np.mean(recall)


def eval_predict_score(netA, netB, mergelayer, disease_drug_data, break_id):

    if next(netA.parameters()).is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    disease_drug_data = disease_drug_data.to(device)

    with torch.no_grad():
        netA.eval()
        netB.eval()

        disease_encoded = netA(disease_drug_data[:, 0:break_id])
        drug_encoded = netB(disease_drug_data[:, break_id:])

        tmp_score = mergelayer(disease_encoded, drug_encoded)
        predict_score = tmp_score

    predict_score = torch.reshape(predict_score, (break_id, -1))

    return predict_score


def eval_predict_score_large_memo(netA, netB, mergelayer, disease_data, drug_data, break_id):
    pairs_num = disease_data.size(0) * drug_data.size(0)
    round_ = pairs_num // break_id
    pairs_idx = flatten_data(disease_data, drug_data, get_idx=True)
    if next(netA.parameters()).is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        netA.eval()
        netB.eval()
        tmp_score_lst = []
        for i in range(round_):

            start_ = break_id * i
            end_ = break_id * (i + 1)

            disease_idx = pairs_idx[0][start_:end_]
            drug_idx = pairs_idx[1][start_:end_]

            disease_encoded = netA(disease_data[disease_idx, :].to(device))
            drug_encoded = netB(drug_data[drug_idx, :].to(device))

            tmp_score = mergelayer(disease_encoded, drug_encoded)
            tmp_score_lst.append(tmp_score)
    
    predict_score = torch.cat(tuple(tmp_score_lst))
    predict_score = torch.reshape(predict_score, (break_id, -1))

    return predict_score


def exec_eval(netA, netB, mergelayer, disease_data, drug_data, Y_train, Y_test):
    break_id = Y_train.size(0)
    if break_id > 1000:
        predict_score = eval_predict_score_large_memo(netA, netB, mergelayer, disease_data, drug_data, break_id)
    else:
        disease_drug_data = flatten_data(disease_data, drug_data)
        predict_score = eval_predict_score(netA, netB, mergelayer, disease_drug_data, break_id)
    if torch.cuda.is_available():
        predict_score = predict_score.cpu()
        Y_train = Y_train.cpu()
        Y_test = Y_test.cpu()

    predict_score = predict_score.detach().numpy()

    Y_train = Y_train.numpy()
    Y_test = Y_test.numpy()

    test_dda_evaluation = DDAEvaluation(Y_train, Y_test, predict_score)
    test_label, test_score = test_dda_evaluation.test_vs_uknw()
    test_roc, test_pr = test_dda_evaluation.auroc_pr(test_label, test_score)


    train_dda_evaluation = DDAEvaluation(Y_test, Y_train, predict_score)
    train_label, train_score = train_dda_evaluation.test_vs_uknw()
    train_roc, train_pr = train_dda_evaluation.auroc_pr(train_label, train_score)

    topN = [10]

    test_map = [test_dda_evaluation.mAP_rec(N)[0] for N in topN]
    test_recall = [test_dda_evaluation.mAP_rec(N)[1] for N in topN]


    train_map = [train_dda_evaluation.mAP_rec(N)[0] for N in topN]
    train_recall = [train_dda_evaluation.mAP_rec(N)[1] for N in topN]

    return test_roc, test_pr, test_map, test_recall, train_roc, train_pr, train_map, train_recall, predict_score



def exec_eval_ensemble(netA_lst, netB_lst, mergelayer, disease_data, drug_data, Y_train, Y_test):
    break_id = Y_train.size(0)
    ensemble_predict_score = 0
    for i in range(len(netA_lst)):
        if break_id > 1000:
            predict_score = eval_predict_score_large_memo(netA_lst[i], netB_lst[i], mergelayer, disease_data, drug_data, break_id)
        else:
            disease_drug_data = flatten_data(disease_data, drug_data)
            predict_score = eval_predict_score(netA_lst[i], netB_lst[i], mergelayer, disease_drug_data, break_id)

        if torch.cuda.is_available():
            predict_score = predict_score.cpu()

        predict_score = predict_score.data.numpy()
        ensemble_predict_score += predict_score


    ensemble_predict_score = ensemble_predict_score / len(netA_lst)

    if torch.cuda.is_available():
        Y_train = Y_train.cpu()
        Y_test = Y_test.cpu()


    Y_train = Y_train.numpy()
    Y_test = Y_test.numpy()

    test_dda_evaluation = DDAEvaluation(Y_train, Y_test, ensemble_predict_score)
    test_label, test_score = test_dda_evaluation.test_vs_uknw()
    test_roc, test_pr = test_dda_evaluation.auroc_pr(test_label, test_score)


    train_dda_evaluation = DDAEvaluation(Y_test, Y_train, ensemble_predict_score)
    train_label, train_score = train_dda_evaluation.test_vs_uknw()
    train_roc, train_pr = train_dda_evaluation.auroc_pr(train_label, train_score)


    topN = [10]

    test_map = [test_dda_evaluation.mAP_rec(N)[0] for N in topN]
    test_recall = [test_dda_evaluation.mAP_rec(N)[1] for N in topN]

    train_map = [train_dda_evaluation.mAP_rec(N)[0] for N in topN]
    train_recall = [train_dda_evaluation.mAP_rec(N)[1] for N in topN]

    return test_roc, test_pr, test_map, test_recall, train_roc, train_pr, train_map, train_recall, ensemble_predict_score



def compute_eval_score(netA, netB, mergelayer, disease_data, drug_data):
    break_id = disease_data.size(0)
    if break_id > 1000:
        predict_score = eval_predict_score_large_memo(netA, netB, mergelayer, disease_data, drug_data, break_id)
    else:
        disease_drug_data = flatten_data(disease_data, drug_data)
        predict_score = eval_predict_score(netA, netB, mergelayer, disease_drug_data, break_id)
    if torch.cuda.is_available():
        predict_score = predict_score.cpu()

    predict_score = predict_score.detach().numpy()
    return predict_score
    
    
def eval_from_score(Y_train, Y_test, predict_score):
    if torch.cuda.is_available():
        if torch.cuda.is_available():
            Y_train = Y_train.cpu()
            Y_test = Y_test.cpu()

    Y_train = Y_train.numpy()
    Y_test = Y_test.numpy()

    test_dda_evaluation = DDAEvaluation(Y_train, Y_test, predict_score)
    test_label, test_score = test_dda_evaluation.test_vs_uknw()
    test_roc, test_pr = test_dda_evaluation.auroc_pr(test_label, test_score)

    train_dda_evaluation = DDAEvaluation(Y_test, Y_train, predict_score)
    train_label, train_score = train_dda_evaluation.test_vs_uknw()
    train_roc, train_pr = train_dda_evaluation.auroc_pr(train_label, train_score)

    topN = [10]

    test_map = [test_dda_evaluation.mAP_rec(N)[0] for N in topN]
    test_recall = [test_dda_evaluation.mAP_rec(N)[1] for N in topN]

    train_map = [train_dda_evaluation.mAP_rec(N)[0] for N in topN]
    train_recall = [train_dda_evaluation.mAP_rec(N)[1] for N in topN]

    return test_roc, test_pr, test_map, test_recall, train_roc, train_pr, train_map, train_recall


