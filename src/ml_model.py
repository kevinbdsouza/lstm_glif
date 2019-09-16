import torch.nn as nn


class SeqParamModel(nn.Module):

    def __init__(self, cfg):
        super(SeqParamModel, self).__init__()
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.param_dim = cfg.param_dim
        self.encode_lstm = nn.LSTM(self.input_dim, self.hidden_dim, 1,
                                   bidirectional=True, batch_first=True)
        self.model_param = nn.Linear(self.hidden_dim * 2, self.param_dim)

    def train_model(self, x):
        encoded_inputs, (h_n, c_n) = self.encode_lstm(x)
        h_n = h_n.view(-1, 1, self.hidden_dim * 2)
        params = self.model_param(h_n)
        return params

    def forward(self, x):
        params = self.train_model(x)

        return params
