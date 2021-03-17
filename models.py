from STAN_WIN.layers import *


class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout=0.1):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.Attn = Attn(emb_l, l_dim-1)

    def forward(self, traj, mat1, mat2, vec, traj_len):
        # long(N, M, [u, l, t]), float(N, M, M, 2), float(L, L), float(N, M), long(N)
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)  # (N, M, emb), (N, M, M, emb)
        self_attn = self.SelfAttn(joint, delta, traj_len)  # (N, M, emb)
        self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)  # (N, M, L, emb)
        output = self.Attn(self_attn, self_delta, traj_len)  # (N, L)
        return output
