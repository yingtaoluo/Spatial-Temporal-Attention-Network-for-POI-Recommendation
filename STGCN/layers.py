import torch
from torch import nn
from torch.nn import functional as F
from ours.utils import sparse_dropout
from torch.autograd import Variable
import random
from torch.nn import init
from torch.nn.utils.rnn import \
    pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import pdb

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'


def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless:  # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support


class SpaAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings and transform
    """
    def __init__(self, id2feat, dis, cuda=False):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(SpaAggregator, self).__init__()

        self.id2feat = id2feat
        self.cuda = cuda
        self.dis = dis

    def forward(self, nodes, to_neighs, num_sample):
        """
        nodes --- list of nodes in a batch
        dis --- shape alike adj
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set  # a disordered non-repeatable list
        # sample neighbors
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = []
            for i, to_neigh in enumerate(to_neighs):
                if len(to_neigh) >= num_sample:
                    samp_neighs.append(_set(_sample(to_neigh, num_sample)))
                elif len(to_neigh) == 0:  # no neigh
                    # print(to_neigh)
                    samp_neighs.append({nodes[i]})
                else:
                    samp_neighs.append(_set(to_neigh))
            # samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample
            #                else _set(to_neigh) for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        # ignore the unlinked nodes
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1  # can be replaced by distance
        # print(torch.sum(torch.isnan(mask)))
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        # spatial_transition = Variable(torch.FloatTensor(len(samp_neighs), len(unique_nodes)))
        # print(unique_nodes_list)
        # pdb.set_trace()
        embed_matrix = self.id2feat(torch.LongTensor(unique_nodes_list))  # （??, feat_dim)
        to_feats = mask.mm(embed_matrix)  # (?, num_sample)
        # print(torch.sum(torch.isnan(embed_matrix)))
        return to_feats  # (?, feat_dim)


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """
    def __init__(self, id2feat, adj_list, dis_list, feature_dim, embed_dim, cuda=False):
        super(SageLayer, self).__init__()

        self.id2feat = id2feat
        self.feat_dim = feature_dim
        self.agg = SpaAggregator(id2feat, dis_list, cuda=cuda)
        self.num_sample = feature_dim
        self.cuda = cuda
        self.adj_list = adj_list
        self.dis_list = dis_list
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.agg(nodes, [self.adj_list[int(node-1)] for node in nodes], self.num_sample)
        if self.cuda:
            self_feats = self.id2feat(torch.LongTensor(nodes).cuda())
        else:
            self_feats = self.id2feat(torch.LongTensor(nodes))
        combined = torch.cat((self_feats, neigh_feats), dim=1)  # (?, 2*feat_dim)
        # print(combined.shape)
        combined = F.relu(self.weight.mm(combined.t()))
        # pdb.set_trace()
        return combined


class RNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, max_len, dropout=0.1):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(max_len, 1, bias=False)
        self.max_len = max_len

    def last(self, out_pad, out_len):
        out = []
        for i, item in enumerate(out_pad):
            if i is 0:
                out = item[out_len[i]-1:out_len[i]]
            else:
                out = torch.cat((out, item[out_len[i]-1:out_len[i]]))
        out = torch.FloatTensor(out).to(device)
        return out

    def forward(self, joint):
        # joint [(?, emb*3), size]
        output, _ = self.rnn(joint)  # [(?, hid_size), size]
        # out_pad (N, loc_max, hid_size), out_len (N)
        out_pad, out_len = pad_packed_sequence(output, batch_first=True)
        [N, loc_max, hid] = out_pad.shape
        rnn_out = self.last(out_pad, out_len)  # (N, hid_size)

        zero_pad = torch.zeros((N, self.max_len - loc_max, hid))
        out_pad = torch.cat((out_pad, zero_pad), dim=1)  # (N, max, hid)

        q, k = self.query(rnn_out), self.key(out_pad)  # (N, hid), (N, max, hid)
        attn = torch.mul(q, k.permute(1, 0, 2))  # (max, N, hid)
        attn = F.softmax(attn, dim=0).permute(1, 2, 0)  # (N, hid, max)
        attn_out = torch.tanh(self.value(attn).view(N, hid))  # (N, hid_size)

        out = rnn_out + attn_out
        return out


class AttnSeq(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, dropout=0.1):
        super(AttnSeq, self).__init__()
        self.linear_q = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.weight)

    def last(self, out_pad, out_len):
        out = []
        for i, item in enumerate(out_pad):
            if i is 0:
                out = item[out_len[i]-1:out_len[i]]
            else:
                out = torch.cat((out, item[out_len[i]-1:out_len[i]]))
        out = torch.FloatTensor(out).to(device)
        return out

    def forward(self, joint):
        # joint [(?, emb*5), size]
        # out_pad (N, max, emb*5), out_len (N)
        return joint


class MultiEmbed(nn.Module):
    def __init__(self, embed_t, embed_l, embed_u, dropout):
        super(MultiEmbed, self).__init__()
        self.embed_t, self.embed_l, self.embed_u = embed_t, embed_l, embed_u

    def forward(self, trajectories):
        # inputs (N, max, [u, l, t])
        trajectories[:, :, 2] = (trajectories[:, :, 2]-1) % 24 + 1  # segment time by 24 hours
        time = self.embed_t(trajectories[:, :, 2])  # (N, max) --> (N, max, embed)
        loc = self.embed_l(trajectories[:, :, 1])  # (N, max) --> (N, max, embed)
        user = self.embed_u(trajectories[:, :, 0])  # (N, max) --> (N, max, embed)
        tl = torch.cat((time, loc), dim=-1)  # (N, max, embed*2)
        tl = torch.cat((tl, user), dim=-1)  # (N, max, embed*3)
        zero = torch.zeros(time.shape)  # (N, max, embed)
        joint = torch.cat((tl, zero), dim=-1)  # (N, max, embed*4)
        joint = torch.cat((joint, zero), dim=-1)  # (N, max, embed*5)
        return joint
