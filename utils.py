import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle


def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["meta_variables"]


def build_domains(train_idx, test_idx, datasets, meta_vars, device):
    domains = []
    test_domains = []
    for idx in train_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                        torch.tensor(X_i, dtype=torch.float32, device=device), torch.Tensor(Y_i).long().to(device)))

    for idx in test_idx:
        X_i, Y_i = datasets[idx]
        meta = meta_vars[idx]
        test_domains.append((torch.tensor([meta], dtype=torch.float32, device=device),
                             torch.tensor(X_i, dtype=torch.float32, device=device),
                             torch.Tensor(Y_i).long().to(device)))
    return domains, test_domains


def build_mlp_layers(input_dim=2, hidden_dim=50, output_dim=1):
    param_shapes = {
        "w1": (hidden_dim, input_dim),
        "b1": (hidden_dim,),
        "w2": (hidden_dim, hidden_dim),
        "b2": (hidden_dim,),
        "w3": (output_dim, hidden_dim),
        "b3": (output_dim,)
    }

    param_size = hidden_dim * input_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim + output_dim * hidden_dim + output_dim
    return param_size, param_shapes


param_size, param_shapes = build_mlp_layers(2, 50, 2)


def forward_with_params(X, flat_params, param_shapes):
    offset = 0
    w1_size = param_shapes["w1"][0] * param_shapes["w1"][1]
    b1_size = param_shapes["b1"][0]
    w2_size = param_shapes["w2"][0] * param_shapes["w2"][1]
    b2_size = param_shapes["b2"][0]
    w3_size = param_shapes["w3"][0] * param_shapes["w3"][1]
    b3_size = param_shapes["b3"][0]

    w1 = flat_params[offset: offset + w1_size]
    offset += w1_size
    b1 = flat_params[offset: offset + b1_size]
    offset += b1_size
    w2 = flat_params[offset: offset + w2_size]
    offset += w2_size
    b2 = flat_params[offset: offset + b2_size]
    offset += b2_size
    w3 = flat_params[offset: offset + w3_size]
    offset += w3_size
    b3 = flat_params[offset: offset + b3_size]
    offset += b3_size

    # 2) Reshape
    w1 = w1.view(param_shapes["w1"])
    b1 = b1.view(param_shapes["b1"])
    w2 = w2.view(param_shapes["w2"])
    b2 = b2.view(param_shapes["b2"])
    w3 = w3.view(param_shapes["w3"])
    b3 = b3.view(param_shapes["b3"])

    # 3) Forward
    hidden = F.linear(X, w1, b1)
    hidden = torch.relu(hidden)
    hidden = F.linear(hidden, w2, b2)
    hidden = torch.relu(hidden)
    out = F.linear(hidden, w3, b3)
    out = torch.log_softmax(out, dim=1)
    return out


predictive_model = nn.Sequential(
    nn.Linear(2, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 2),
    nn.LogSoftmax(dim=1)
)

backbone_model = nn.Sequential()


def generalized_model(domain_x, domain_param):
    weights = {}
    biases = {}
    start_idx = 0
    for name, p in predictive_model[0].state_dict().items():
        end_idx = start_idx + p.numel()
        if name.endswith("bias"):
            biases[name] = domain_param[start_idx:end_idx].view(p.shape)
        elif name.endswith("weight"):
            weights[name] = domain_param[start_idx:end_idx].view(p.shape)
        else:
            raise ValueError('Not defined layer!')
        start_idx = end_idx

    x = domain_x
    for name, layer in predictive_model[0].named_children():
        if isinstance(layer, nn.Linear):
            weight_name = f"{name}.weight"
            bias_name = f"{name}.bias"
            x = F.linear(x, weights[weight_name], biases[bias_name])
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)
        elif isinstance(layer, nn.Sigmoid):
            x = torch.sigmoid(x)
        elif isinstance(layer, nn.LogSoftmax):
            x = F.log_softmax(x, dim=layer.dim)
        elif isinstance(layer, nn.Dropout):
            pass
        else:
            raise ValueError('Not defined layer!')

    domain_y = x
    return domain_y


class Encoder(nn.Module):
    def __init__(self, param_dim, embed_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim))

    def forward(self, flat_params):
        return self.net(flat_params)


class Decoder(nn.Module):
    def __init__(self, embed_dim=8, param_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, param_dim))

    def forward(self, E):
        return self.net(E)
