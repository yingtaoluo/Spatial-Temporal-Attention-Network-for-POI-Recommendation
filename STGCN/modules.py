from ours.layers import *


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, dropout, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(GraphConvolution(self.input_dim, hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=dropout,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=dropout,
                                                     is_sparse_inputs=False),
                                    )

    def forward(self, inputs):
        x, support = inputs
        x = self.layers((x, support))

        return x

    def l2_loss(self):
        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss


class UppGraph(nn.Module):
    def __init__(self, id2feat, feature_dim, embed_dim, cuda=False):
        super(UppGraph, self).__init__()
        self.id2feat = id2feat
        self.cuda = cuda
        self.num_sample = 1
        self.Linear = nn.Linear(self.num_sample*feature_dim, embed_dim)
        self.feature_dim = feature_dim

    def forward(self, regions):
        # regions (?, *num, *point), region (*num, *point), sample_neighs (?, num, *point)
        _sample = random.sample
        _set = set
        sample_regions = []  # (?, *num, feat_dim)
        for region in regions:
            sample_region = []
            for reg in region:
                if len(reg) >= self.feature_dim:
                    sample_region.append(_set(_sample(reg, self.feature_dim)))
                else:
                    sample_region.append(_set(reg))
            if len(region) == 0:
                inner = [0] * self.feature_dim
                sample_region = [{inn} for inn in inner]*self.num_sample

            unique_nodes_list = list(set.union(*sample_region))
            unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
            mask = Variable(torch.zeros(len(sample_region), len(unique_nodes)))  # (*num, ??)
            column_indices = [unique_nodes[n] for samp_neigh in sample_region for n in samp_neigh]
            row_indices = [i for i in range(len(sample_region)) for j in range(len(sample_region[i]))]
            mask[row_indices, column_indices] = 1  # can be replaced by distance
            # print(torch.sum(torch.isnan(mask)))
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh)
            # spatial_transition = Variable(torch.FloatTensor(len(samp_neighs), len(unique_nodes)))
            embed_matrix = self.id2feat(torch.LongTensor(unique_nodes_list))  # （??, feat_dim)
            to_feats = mask.mm(embed_matrix)  # (*num, feat_dim)
            sample_regions.append(to_npy(to_feats).tolist())

        sample_neighs = []  # (?, num, feat_dim)
        for region in sample_regions:
            if len(region) > self.num_sample:
                sample_neighs.append(_sample(region, self.num_sample))
            elif len(region) == 0:
                sample_neighs.append([[0]*self.feature_dim]*self.num_sample)
            else:
                sample_neighs.append(_sample(np.array(region).repeat
                (self.num_sample, axis=0).tolist(), self.num_sample))

        hi_emb = torch.FloatTensor(sample_neighs).view(len(regions), -1)  # (?, num * feat_dim)
        to_emb = self.Linear(hi_emb)  # (?, emb)
        return to_emb


class HiGraph(nn.Module):
    def __init__(self, id2feat, group, feature_dim, embed_dim, u_dim):
        super(HiGraph, self).__init__()
        self.group = group  # (user, *num, *point)
        self.layer = UppGraph(id2feat, feature_dim, embed_dim)

    def forward(self, users):
        # users (?) start from 1
        batch_group = list_slice_by_index(self.group, users)  # (?, *num, *point)
        region_embeds = self.layer(batch_group)  # (?, emb)
        return region_embeds


class GraphSage(nn.Module):
    def __init__(self, num_node, feature_dim, embed_dim, adj, dis, poi, group, u_dim):
        super(GraphSage, self).__init__()
        id2node = nn.Embedding(num_node, feature_dim)

        layer1 = SageLayer(id2node, adj, dis, feature_dim, embed_dim)
        layer12 = SageLayer(lambda nodes: layer1(nodes).t(), adj, dis, embed_dim, embed_dim)
        self.transition = layer12
        self.group = HiGraph(id2node, group, feature_dim, embed_dim, u_dim)

    def forward(self, nodes, users):
        neigh_embeds = self.transition(nodes).t()  # (?, emb)
        region_embeds = self.group(users)  # (?, emb)
        return torch.cat((neigh_embeds, region_embeds), dim=-1)  # (?, emb*2)


def list_slice_by_index(MyList, Indexs):
    slices = []
    for i in Indexs:
        slices.append(MyList[i-1])
    return slices

