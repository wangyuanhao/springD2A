from model import *
import matplotlib as mpl
mpl.use("agg")
from untils import *
from evaluation import *
from data_loader import CVDataLoader
from loss import *
import pandas as pd
from copy import deepcopy
import numpy as np


def print_evaluation(logger, epoch, netA, netB, mergelayer, disease_data, drug_data, Y_train_interact, Y_test_interact, predict_score=None):

    if predict_score is not None:
        eval_res = eval_from_score(Y_train_interact, Y_test_interact, predict_score)
    else:
        eval_res = exec_eval(netA, netB, mergelayer, disease_data, drug_data, Y_train_interact, Y_test_interact)
        predict_score = eval_res[8]

    test_roc, test_pr = eval_res[0], eval_res[1]
    test_map = eval_res[2]
    test_recall = eval_res[3]
    test_map10 = test_map[0]
    test_recall10 = test_recall[0]


    train_roc, train_pr = eval_res[4], eval_res[5]
    train_map = eval_res[6]
    train_recall = eval_res[7]
    train_map10 = train_map[0]
    train_recall10 = train_recall[0]
    train_recommend_record = "In %d epoch, in train, AUROC:%.5f, AUPR:%.5f, mAP@10:%.5f, Rec@10:%.5f" \
                             % (epoch + 1, train_roc, train_pr, train_map10, train_recall10)
    logger.info(train_recommend_record)
    test_recommend_record = "In %d epoch, in test, AUROC:%.5f, AUPR:%.5f, mAP@10:%.5f, Rec@10:%.5f" \
                            % (epoch + 1, test_roc, test_pr, test_map10, test_recall10)
    logger.info(test_recommend_record)

    
    return train_roc, train_pr, test_roc, test_pr, predict_score, logger


def score_recorder(netA, netB, mergelayer, disease_data, drug_data, score):
    predicit_score = compute_eval_score(netA, netB, mergelayer, disease_data, drug_data)
    score += predicit_score
    return score


def score_average(score, count):
    return score / count


def model_recorder(net, recorder):
    with torch.no_grad():
        for name, param in net.named_parameters():
            recorder[name] += param.data
    return recorder


def initial_model_recorder(net):
    recorder = dict()
    for name, param in net.named_parameters():
        recorder[name] = 0

    return recorder


def model_average(net, count, recorder, device):
    with torch.no_grad():
        ensemble_model = type(net)(net.inputs, net.bottleneck, net.dropout)
        ensemble_model = ensemble_model.to(device)

        for name, param in ensemble_model.named_parameters():
            param.data = recorder[name] / count

    return ensemble_model


def initial_params(net, xi, device):
    new_model = FeatureExtractor(net.inputs, net.bottleneck, net.dropout)
    new_model = new_model.float()
    new_model = new_model.to(device)

    for new_param, param in zip(new_model.parameters(), net.parameters()):
        new_param.data.mul_(1 - xi).add_(xi, param.data)

    return new_model


def model_agregation(net_lst, device):

    with torch.no_grad():
        net_zero = net_lst[0]

        params_dict = dict()
        for name, param in net_zero.named_parameters():
            params_dict[name] = 0

        for net in net_lst:
            for name, param in net.named_parameters():
                params_dict[name] += param.data

        ensemble_model = type(net_zero)(net_zero.inputs, net_zero.bottleneck, net_zero.dropout)
        ensemble_model = ensemble_model.to(device)

        for name, param in ensemble_model.named_parameters():
            param.data = params_dict[name] / len(net_lst)

    return ensemble_model


def zero_out_gradient(net):
    for param in net.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
    return net

def update_boundary(all_unkwn_idx, on_unkwn_idx, all_unkwn_score, gamma):
    res_all_unkwn_bool = np.array([True if i not in on_unkwn_idx else False for i in list(all_unkwn_idx)])
    res_all_unkwn_idx = all_unkwn_idx[res_all_unkwn_bool]
    res_all_unkwn_score = all_unkwn_score[res_all_unkwn_bool]


    up_margin = gamma
    res_unkwn_boundary_bool = np.array([score_ < up_margin for score_ in list(res_all_unkwn_score)])
    res_unkwn_boundary_idx = res_all_unkwn_idx[res_unkwn_boundary_bool]


    """propotional weight"""
    res_unkwn_boundary_score = res_all_unkwn_score[res_unkwn_boundary_bool]
    norm_res_unkwn_boundary_score = res_unkwn_boundary_score / res_unkwn_boundary_score.sum()

    """constant weight"""
    # norm_res_unkwn_boundary_score = np.ones((len(res_unkwn_boundary_idx),)) / len(res_unkwn_boundary_idx)

    """reversed weigth"""
    # res_unkwn_boundary_score = 1 - res_all_unkwn_score[res_unkwn_boundary_bool]
    # norm_res_unkwn_boundary_score = res_unkwn_boundary_score / res_unkwn_boundary_score.sum()

    return res_unkwn_boundary_idx, norm_res_unkwn_boundary_score


def train(logger, netA, netB, paramergelayer, pos_X_train_idx, unkwn_pairs_idx,
          disease_data, drug_data, Y_train_interact, Y_test_interact,
          gamma,  xi, freq, rho, num_epochs, batch_size, lr, device, seed):

    train_ce_ls, test_ce_ls = [], []
    train_roc_ls, train_pr_ls, test_roc_ls, test_pr_ls = [], [], [], []
    

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    paramergelayer = paramergelayer.float()
    paramergelayer = paramergelayer.to(device)


    Y_train_interact = Y_train_interact.to(device)
    Y_test_interact = Y_test_interact.to(device)


    logger.info("training on "+str(device))
    all_unkwn_pairs_idx = deepcopy(unkwn_pairs_idx)

    netA = netA.float()
    netA = netA.to(device)
    netB = netB.float()
    netB = netB.to(device)

    params = [
        {"params": netA.parameters()},
        {"params": netB.parameters()},
    ]
    optimizer = torch.optim.Adam(params=params, lr=lr)

    # record models
    netA_recorder = initial_model_recorder(netA)
    netB_recorder = initial_model_recorder(netB)
    score_recorder_ = 0
    
    neg_sample_weight = np.ones((len(unkwn_pairs_idx), )) / len(unkwn_pairs_idx)
    update_count = 1
    for epoch in range(num_epochs):

        train_iter, select_unkwn_pairs_idx = data_loader(pos_X_train_idx, unkwn_pairs_idx, batch_size,
                                                         rho, neg_sample_weight)

        train_ls_sum, train_acc_sum, n, batch_count = 0, 0, 0, 0

        for i, item in enumerate(train_iter):
            idx = torch.cat((item[0][0], item[1][0]), dim=0)
            y = torch.cat((item[0][1], item[1][1]), dim=0)
            y = y.to(device)
            row_idx, col_idx = ind2sub((disease_data.shape[0], drug_data.shape[0]), idx)

            disease_encoded = netA(disease_data.to(device))
            drug_encoded = netB(drug_data.to(device))

            batch_disease_data = disease_encoded[row_idx, :]
            batch_drug_data = drug_encoded[col_idx, :]

            mlp_out = paramergelayer(batch_disease_data, batch_drug_data, y)

            optimizer.zero_grad()
            loss = dda_loss(mlp_out.view(-1, ), y)


            loss.backward()
            optimizer.step()

            train_ls_sum += loss.item()
            train_acc_sum += ((mlp_out.view(-1, ) > 0.5).float() == y).sum().item()
            n += y.shape[0]
            batch_count += 1

        if ((epoch + 1) % 10) == 0:
            train_ce = train_ls_sum / batch_count
            train_ce_ls.append(train_ce)

            train_record = "In %d epoch, train loss:%.5f" % (epoch + 1, train_ce_ls[-1])
            logger.info(train_record)

            train_roc, train_pr, test_roc, test_pr, _, logger = print_evaluation(logger, epoch, netA, netB, paramergelayer, disease_data,
                                                                              drug_data, Y_train_interact, Y_test_interact)
            train_roc_ls.append(train_roc)
            train_pr_ls.append(train_pr)
            test_roc_ls.append(test_roc)
            test_pr_ls.append(test_pr)

        if update_count <= 2:
            update_freq = 200
        else:
            update_freq = freq
        # change data and model
        if ((epoch + 1) % update_freq) == 0:
            logger.info("update frequency: %d" % update_freq)
            with torch.no_grad():

                # update negative sequence
                # get all score of unknown pairs
                unkwn_row_idx, unkwn_col_idx = ind2sub((disease_data.shape[0], drug_data.shape[0]), all_unkwn_pairs_idx)
                unkwn_disease_encoded = disease_encoded[unkwn_row_idx, :].data
                unkwn_drug_encoded = drug_encoded[unkwn_col_idx, :].data
                all_unkwn_score = paramergelayer(unkwn_disease_encoded, unkwn_drug_encoded).view(-1, )
                all_unkwn_score = deepcopy(all_unkwn_score.data.cpu().numpy())



                unkwn_pairs_idx, neg_sample_weight = update_boundary(all_unkwn_pairs_idx, select_unkwn_pairs_idx, all_unkwn_score, gamma)


            # record models
            netA_recorder = model_recorder(netA, netA_recorder)
            netB_recorder = model_recorder(netB, netB_recorder)
            score_recorder_ = score_recorder(netA, netB, paramergelayer, disease_data, drug_data, score_recorder_)
            
            # update model with smoothing initialization
            netA = initial_params(netA, xi, device)
            netB = initial_params(netB, xi, device)

            # update learning parameters and optimizer
            params = [
                {"params": netA.parameters()},
                {"params": netB.parameters()},
            ]
            optimizer = torch.optim.Adam(params=params, lr=lr)
            update_count += 1

        if (epoch + 1) == num_epochs:
            ensembleA = model_average(netA, update_count, netA_recorder, device)
            ensembleB = model_average(netB, update_count, netB_recorder, device)
            ensemble_score = score_average(score_recorder_, update_count)

            logger.info("In ensemble model parameters(#%d)" % update_count)
            _, _, _, _, params_predict_score, logger = print_evaluation(logger, epoch, ensembleA, ensembleB, paramergelayer, disease_data,
                                                  drug_data, Y_train_interact, Y_test_interact)

            logger.info("In ensemble model outputs(#%d)" % update_count)
            _, _, _, _, pred_predict_score, logger = print_evaluation(logger, epoch, None, None, None, disease_data,
                                                  drug_data, Y_train_interact, Y_test_interact, predict_score=ensemble_score)

    return train_ce_ls, train_roc_ls, train_pr_ls, test_ce_ls, test_roc_ls, test_pr_ls, logger


def kfold_cv(logger, kfold, disease_fi, drug_fi, interact_fi,
             gamma, alpha, beta, xi, freq, rho,
             num_epochs, batch_size, lr, device, seed):

    cv = CVDataLoader(disease_fi, drug_fi, interact_fi)
    drug_data = cv.typeB_data
    disease_data = cv.typeA_data

    drug_data = torch.tensor(drug_data, dtype=torch.float)
    disease_data = torch.tensor(disease_data, dtype=torch.float)

    disease_size = disease_data.shape[0]
    drug_size = drug_data.shape[0]
    fold_count = 1
    for cv_return in cv.cross_validation_iter(kfold):


        logger.info("*"*20 + "In %d-th fold" % fold_count + "*"*20)

        pos_X_train_idx, pos_X_test_idx = cv_return[0], cv_return[1]
        unkwn_pairs_idx = cv_return[2]
        Y_train_interact, Y_test_interact = cv_return[3], cv_return[4]

        Y_train_interact = torch.tensor(Y_train_interact, dtype=torch.float)
        Y_test_interact = torch.tensor(Y_test_interact, dtype=torch.float)

        data = (pos_X_train_idx, unkwn_pairs_idx, disease_data, drug_data, Y_train_interact, Y_test_interact)

        netA = FeatureExtractor(disease_size, 128)
        netB = FeatureExtractor(drug_size, 128)

        paramergelayer = ParamMergeLayer(128, 128, alpha, beta)
        train_ce_ls, train_roc_ls, train_pr_ls, test_ce_ls, test_roc_ls, test_pr_ls, logger = train(logger, netA, netB,
                                                                                                    paramergelayer, *data,
                                                                                                    gamma,  xi, freq, rho,
                                                                                                    num_epochs, batch_size,
                                                                                                    lr, device, seed)
        fold_count += 1


def denovo_test(logger, kfold, disease_fi, drug_fi, interact_fi,
             gamma, alpha, beta, xi, freq, rho,
             num_epochs, batch_size, lr, device, seed):

    disease_data = pd.read_csv(disease_fi, header=None, index_col=None)
    drug_data = pd.read_csv(drug_fi, header=None, index_col=None)

    disease_size = disease_data.shape[0]
    drug_size = drug_data.shape[0]

    interact = pd.read_csv(interact_fi, header=None, index_col=None)
    drug_data = torch.tensor(drug_data.values, dtype=torch.float)
    disease_data = torch.tensor(disease_data.values, dtype=torch.float)

    interact = interact.values
    denovo_drug = np.where(interact.sum(axis=0) == 1)[0]
    Y_train_interact = deepcopy(interact)
    Y_train_interact[:, denovo_drug] = 0
    Y_test_interact = interact - Y_train_interact

    unkwn_pairs_idx = np.where(interact.reshape(-1, ) == 0)[0]
    pos_X_train_idx = np.where(Y_train_interact.reshape(-1, ) == 1)[0]

    Y_train_interact = torch.tensor(Y_train_interact, dtype=torch.float)
    Y_test_interact = torch.tensor(Y_test_interact, dtype=torch.float)

    data = (pos_X_train_idx, unkwn_pairs_idx, disease_data, drug_data, Y_train_interact, Y_test_interact)



    netA = FeatureExtractor(disease_size, 128)
    netB = FeatureExtractor(drug_size, 128)

    paramergelayer = ParamMergeLayer(128, 128, alpha, beta)
    train(logger, netA, netB, paramergelayer, *data, gamma, xi, freq, rho, num_epochs, batch_size, lr, device, seed)


