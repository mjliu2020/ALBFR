import numpy as np
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")
import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange


class AE_Encoder(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, latent_size):
        super(AE_Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear1(x)) #-> bs,hidden_size
        x = self.linear2(x) #-> bs,latent_size
        return x


class AE_Decoder(nn.Module):

    def __init__(self, d_emd=175, d_vol=8):
    
        super(AE_Decoder, self).__init__()
        self.de_emd = nn.Linear(d_emd, d_vol)
        
    def forward(self, x):
        return torch.tanh(self.de_emd(x))


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
        
        
class CNN_Encoder(nn.Module):
    def __init__(self, latentsize):
        super().__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        self.main = nn.Sequential(
            init_(nn.Conv1d(1, 32, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv1d(32, 64, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 64, 3, stride=2)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 10, 1, stride=1)),
            nn.ReLU(),
            Flatten(),
        )

        self.linear = init_(nn.Linear(410, latentsize))

    def forward(self, x):

        feature = self.main(x)
        out = self.linear(feature)
        out = F.relu(out)

        return out
        
        
class CNN_Decoder(nn.Module):
    def __init__(self, latentsize):
        super().__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        self.main = nn.Sequential(
            init_(nn.Conv1d(1, 32, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv1d(32, 64, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 64, 3, stride=2)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 10, 1, stride=1)),
            nn.ReLU(),
            Flatten(),
        )

        self.linear = init_(nn.Linear(410, latentsize))

    def forward(self, x):

        batchsize = x.shape[0]

        feature = self.main(x)
        out = self.linear(feature)

        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'
    query: (nbatches, seq_len, d_k)
    key:   (nbatches, seq_len, d_k)
    value: (nbatches, seq_len, d_v)

    For multi-head attention:
    query: (nbatches, h, seq_len, d_k)
    key:   (nbatches, h, seq_len, d_k)
    value: (nbatches, h, seq_len, d_k)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Self_Attention_Layer(nn.Module):
    def __init__(self, d_k, d_v):
        "Take in embedding size and number of heads."
        super(Self_Attention_Layer, self).__init__()

        self.linears = clones(nn.Linear(d_k, d_v), 3)
        self.attn = None

    def forward(self, query, key, value):
        nbatches = query.size(0)

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value)

        return x
        
        
class Attention_Layer(nn.Module):
    def __init__(self, d_k, d_v):
        "Take in embedding size and number of heads."
        super(Attention_Layer, self).__init__()
        # 3 for Q,K,V
        self.linear_q = nn.Linear(16, d_k)
        self.linear_k = nn.Linear(16, d_k)
        self.linear_v = nn.Linear(16, d_v)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        x, self.attn = attention(query, key, value)

        return x


class RelationModel(torch.nn.Module):

    def __init__(self, input_size, output_size, latent_size, relation_size):
        super(RelationModel, self).__init__()

        self.sigEncoder = CNN_Encoder(latent_size)
        self.w = nn.Linear(latent_size, latent_size)
        self.relation = nn.Linear(latent_size * 2, relation_size)

        self.linear = nn.Linear(relation_size, output_size)
        self.decoder = AE_Decoder(output_size, output_size)

        self.gpttran = nn.Linear(relation_size, 1)

        self.posi = nn.Linear(3, latent_size)
        
    def forward(self, x1, x2, posi1, posi2): #x: bs,input_size

        xr1 = x1.reshape(-1, 1, 175)
        xr2 = x2.reshape(-1, 1, 175)

        sigEmb1 = self.sigEncoder(xr1)
        sigEmb2 = self.sigEncoder(xr2)

        sigEmb1 = sigEmb1 + self.posi(posi1)
        sigEmb2 = sigEmb2 + self.posi(posi2)

        feature1 = self.w(sigEmb1)
        feature2 = self.w(sigEmb2)
        feautre = torch.cat((feature1, feature2), dim = 1)
        relation = F.tanh(self.relation(feautre))
        constrict = self.gpttran(relation)

        relation_linear = F.tanh(self.linear(relation))
        re_x1 = self.decoder(relation_linear + x2)
        re_x2 = self.decoder(relation_linear + x1) 

        return re_x1.reshape(-1, 175), re_x2.reshape(-1, 175), relation, constrict
