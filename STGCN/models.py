from ours.modules import *


class Model(nn.Module):
    def __init__(self, num_node, num_feat, t_dim, l_dim, u_dim, max_len,
                 embed_dim, num_layer, hidden_dim, adj, dis, poi, g, dropout=0.1):
        super(Model, self).__init__()
        self.embed_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        self.embed_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        self.embed_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        self.GraphSage = GraphSage(num_node=num_node, feature_dim=num_feat, embed_dim=embed_dim,
                                   adj=adj, dis=dis, poi=poi, group=g, u_dim=u_dim)
        self.MultiEmbed = MultiEmbed(self.embed_t, self.embed_l, self.embed_u, dropout)
        self.RNN = RNN(num_layer, hidden_dim, embed_dim*5, max_len, dropout)
        self.OutLayer = nn.Linear(hidden_dim, l_dim)
        self.node_len = len(adj)
        self.emb_dim = embed_dim

    def forward(self, traj, traj_len):  # (N, max_seq_len, [u, l, t]), N
        # user_id = traj[:, 0, 0]  # (N)
        # multi-modal embedding layer
        joint = self.MultiEmbed(traj)  # (N, max, emb*5)
        joint_packed = pack_padded_sequence(joint, traj_len, batch_first=True)  # [(?, emb*5), size]
        traj_packed = pack_padded_sequence(traj, traj_len, batch_first=True)  # [(?, [u, l, t]), size]
        loc_nodes = traj_packed[0][:, 1]  # (?)
        loc_users = traj_packed[0][:, 0]  # (?)
        # spatial graph module
        g = self.GraphSage(loc_nodes, loc_users)  # (?, emb*2)
        joint_packed[0][:, self.emb_dim*3:] = g  # incorporate tl with g
        # temporal recurrent module
        h = self.RNN(joint_packed)  # (N, hid_size)
        # output layer
        output = self.OutLayer(h)  # (N, l_dim)
        return output
