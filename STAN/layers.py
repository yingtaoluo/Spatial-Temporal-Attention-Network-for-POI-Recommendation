import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import random
from torch.nn import init
from torch.nn.utils.rnn import \
    pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import pdb

seed = 0
hours = 24*7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'


def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


class Attn(nn.Module):
    def __init__(self, emb_size, dropout=0.1):
        super(Attn, self).__init__()
        self.value = nn.Linear(emb_size, 1, bias=False)

    def forward(self, self_attn, self_delta, traj_len):
        self_delta = torch.sum(self_delta, -1)  # squeeze the embed dimension
        [N, M, L] = self_delta.shape
        # self_attn (N, M, emb), self_delta (N, M, L), len [N]
        attn = torch.bmm(self_delta.transpose(-1, -2), self_attn)  # (N, L, emb)
        attn_out = self.value(attn).view(N, L)  # (N, L)
        # attn_out = F.log_softmax(attn_out, dim=-1)  # ignore if cross_entropy_loss

        return attn_out  # (N, L)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)
        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


class Embed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj_loc, mat2, vec, traj_len, l_max):
        # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
        delta_t = vec.unsqueeze(-1).expand(-1, -1, l_max)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):  # N
            mask[i, 0:traj_len[i]] = 1
            delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i]-1)[:traj_len[i]])

        # pdb.set_trace()

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_s - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, L, emb)

        return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, \
        self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(traj[:, :, 2])  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj[:, :, 1])  # (N, M) --> (N, M, embed)
        user = self.emb_u(traj[:, :, 0])  # (N, M) --> (N, M, embed)
        joint = time + loc + user  # (N, M, embed)

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]  # (N, M, M)
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_s - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl*vsu+esu*vsl) / (self.su-self.sl)
        time_interval = (etl*vtu+etu*vtl) / (self.tu-self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta
