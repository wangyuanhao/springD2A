import torch
from torch import nn
from collections import OrderedDict


class FeatureExtractor(nn.Module):

    def __init__(self, inputs, bottleneck, dropout=0):
        super(FeatureExtractor, self).__init__()
        self.inputs = inputs
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.encoder = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(inputs, inputs*2)),
            ("relu1", nn.ReLU()),
            ("dropuout", nn.Dropout(dropout)),
            ("linear2", nn.Linear(inputs*2, 512)),
            ("relu2", nn.ReLU()),
            ("linear3", nn.Linear(512, bottleneck))
        ]))

    def forward(self, x):
        encoded = self.encoder(x)
        norm_encoded = encoded / torch.norm(encoded, dim=1, keepdim=True)
        return norm_encoded


class ParamMergeLayer(nn.Module):

    def __init__(self, disease_input, drug_input, alpha, beta):

        super(ParamMergeLayer, self).__init__()

        if drug_input != disease_input:
            assert "The dimension of disease and drug should be equal!"
        self.alpha = alpha
        self.gamma = beta

    def forward(self, disease_emb, drug_emb, y=None):
        hadamard_emb = torch.mul(disease_emb, drug_emb).sum(dim=1)
        alpha = self.alpha
        beta = self.beta
        if y is None:
            x2 = hadamard_emb / 0.1
        else:
            x2 = hadamard_emb / ((alpha ** y) * (beta ** (1-y)))
        x3 = torch.sigmoid(x2)

        return x3


