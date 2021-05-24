import matplotlib as mpl
mpl.use("agg")
from springD2A import *
from untils import get_logger
import argparse


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, action="store", dest="data_path", help="data path")
    parser.add_argument("--d", type=str, action="store", dest="dataset", help="data set")
    parser.add_argument("--log", type=str, action="store", dest="log_fi", help="log file")

    parser.add_argument("--gamma", type=float, action="store", dest="gamma", default=0.9,
                        help="gamma")
    parser.add_argument("--alpha", type=float, action="store", dest="alpha", default=0.2,
                        help="alpha in the bias scoring layer")
    parser.add_argument("--beta", type=float, action="store", dest="beta", default=0.05,
                        help="beta in the bias scoring layer")
    parser.add_argument("--xi", type=float, action="store", dest="xi", default=0.99,
                        help="xi in smoothing initialization")
    parser.add_argument("--freq", type=float, action="store", default=10, dest="freq",
                        help="update frequency of sequential sampling")
    parser.add_argument("--rho", type=int, action="store", default=5, dest="rho",
                        help="positive-negative ratio")

    parser.add_argument("--epoch", type=int, action="store", default=1000, dest="num_epochs",
                        help="number of epochs in training")
    parser.add_argument("--bs", type=int, action="store", default=128, dest="batch_size",
                        help="batch size")
    parser.add_argument("--lr", type=float, action="store", default=1e-4, dest="lr",
                        help="learning rate of optimizer")
    parser.add_argument("--kfold", type=int, action="store", default=10, dest="kfold",
                        help="the number folds in cross-validation")

    return parser


if __name__=="__main__":
    parser = params()
    args = parser.parse_args()

    data_path = args.data_path
    dataset = args.dataset
    log_fi = args.log_fi

    gamma = args.gamma
    alpha = args.alpha
    beta = args.beta
    xi = args.xi
    freq = args.freq
    rho = args.rho

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    kfold = args.kfold

    disease_fi = data_path + "/" + dataset + "/" + "disease_sim.csv"
    drug_fi = data_path + "/" + dataset + "/" + "drug_sim.csv"
    interact_fi = data_path + "/" + dataset + "/" + "disease_drug_interact.csv"


    seed = 123
    logger = get_logger(log_fi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    parameter_settings = (gamma, alpha, beta, xi, freq, rho, num_epochs, batch_size, lr)
    kfold_cv(logger, kfold, disease_fi, drug_fi, interact_fi, *parameter_settings, device, seed)


