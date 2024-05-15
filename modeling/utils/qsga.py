import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from .convit import Mlp


class QSGA(nn.Module):
    def __init__(self,
                 cfg,
                 in_dim,
                 out_dim,
                 average_support_first,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 locality_strength=1.,
                 use_local_init=True):
        super().__init__()

        self.cfg = cfg
        self.average_support_first = average_support_first

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.k = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, out_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        if self.cfg.CONVIT_POS:
            self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        else:
            self.gating_param = torch.ones(self.num_heads).cuda() * -100
        # self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init()
            # self.local_init(dtype=torch.float16 if self.cfg.AMP.ACTIVATED else torch.float32)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q_feat, s_feat, q_shapes, s_shapes):
        self.q_shapes = q_shapes
        self.s_shapes = s_shapes

        B, Nq, _ = q_feat.shape
        M, Ns, _ = s_feat.shape
        C = self.out_dim

        if not hasattr(self, 'rel_indices') or self.rel_indices.size(
                1) != Nq or self.rel_indices.size(2) != Ns:
            self.get_rel_indices(Nq, Ns)

        attn = self.get_attention(q_feat, s_feat)
        # print(attn.mean())
        v = self.v(s_feat).reshape(M, Ns, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)

        v = v.unsqueeze(0).repeat(B, 1, 1, 1,
                                  1).flatten(end_dim=1)  #BM, N_h, Ns, C // N_h
        query_aligned_features = attn @ v #BM, Nh, Nq, C//Nh
        query_aligned_features = query_aligned_features.transpose(
            1, 2).flatten(start_dim=-2).reshape(B, M, Nq, C)
        query_aligned_features = self.proj(query_aligned_features)
        query_aligned_features = self.proj_drop(query_aligned_features)
        return query_aligned_features

    def get_attention(self, q_feat, s_feat):
        B, Nq, _ = q_feat.shape
        M, Ns, _ = s_feat.shape

        C = self.out_dim

        q = self.q(q_feat).reshape(B, Nq, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(s_feat).reshape(M, Ns, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)

        pos_score = self.rel_indices

        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)  #BM, N_h, N_q, N_s


        patch_score = torch.einsum('bhqc,mhsc->bmhqs', q,
                                   k).flatten(end_dim=1) * self.scale

        q = q.unsqueeze(1).repeat(1, M, 1, 1, 1)
        k = k.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        patch_score = q@k.permute(0,1,2,4,3)
        patch_score = patch_score.flatten(end_dim=1) * self.scale

        patch_score = patch_score.softmax(dim=-1)

        pos_score = pos_score.softmax(dim=-1)  #BM, N_q, N_s, 3

        gating = torch.sigmoid(self.gating_param.view(1, -1, 1, 1))
        # gating = 0
        attn = (1. - gating) * patch_score + gating * pos_score

        attn /= attn.sum(dim=-1).unsqueeze(-1)  # normalize
        # attn = patch_score

        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, q_feat, s_feat, return_map=True):
        B, Nq, _ = q_feat.shape
        M, Ns, _ = s_feat.shape
        C = self.out_dim
        K = self.cfg.FEWSHOT.K_SHOT
        # for non-locality measurement
        #TO DO change to deal with different support separately
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(
                1) != Nq or self.rel_indices.size(2) != Ns:
            self.get_rel_indices(Nq, Ns)

        if self.average_support_first:
            # if support is already averaged
            attn_map = self.get_attention(q_feat, s_feat) # B, N_ways, Nh, Nq, Ns
        else:
            attn_map = self.get_attention(q_feat, s_feat).reshape(B, M // K, K, -1, Nq, Ns)
            attn_map = attn_map.mean(2)  # average over support

        
        distances = self.rel_indices.squeeze()[:, :, -1]**.5
        dist = torch.einsum('nm,bwhnm->bwh', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, min_alpha=0.002, max_alpha=0.2, dtype=torch.float32):
        self.v.weight.data.copy_(torch.eye(self.out_dim, self.in_dim, dtype=dtype))
        N_h = self.pos_proj.weight.data.shape[0]
        alphas = torch.tensor([
            min_alpha + (max_alpha - min_alpha) / (N_h  )**2.5 * i**2.5
            for i in range(N_h)
        ], dtype=dtype)

        self.pos_proj.weight.data = torch.zeros_like(self.pos_proj.weight.data, dtype=dtype)
        self.pos_proj.weight.data[:, 0] = -alphas

    def get_rel_indices_sq(self, n_q, n_s):
        device = self.q.weight.device
        dtype = torch.float16 if self.cfg.AMP.ACTIVATED else torch.float32
        rel_indices = torch.zeros((n_q, n_s, 3), device=device, dtype=dtype)

        offset_h, offset_w = 0, 0
        for level_w, w in enumerate(self.q_shapes):
            offset_h = 0
            w = w[0]
            for level_h, h in enumerate(self.s_shapes):
                h = h[0]
                ind = torch.arange(h).view(1, -1) / h - torch.arange(w).view(
                    -1, 1) / w
                indx = ind.repeat(w, h) * math.sqrt(w * h)
                indy = ind.repeat_interleave(w, dim=0).repeat_interleave(
                    h, dim=1) * math.sqrt(w * h)
                indd = indx**2 + indy**2

                rel_indices[offset_w:offset_w + w**2, offset_h:offset_h + h**2,
                            0] = indd
                rel_indices[offset_w:offset_w + w**2, offset_h:offset_h + h**2,
                            1] = indx
                rel_indices[offset_w:offset_w + w**2, offset_h:offset_h + h**2,
                            2] = indy
                offset_h += h**2
            offset_w += w**2

        self.rel_indices = rel_indices.unsqueeze(0)

    def get_rel_indices(self, n_q, n_s):
        device = self.q.weight.device
        dtype = torch.float16 if self.cfg.AMP.ACTIVATED else torch.float32
        rel_indices = torch.zeros((n_q, n_s, 3), device=device, dtype=dtype)

        offset_q, offset_s = 0, 0
        for level_q, (hq, wq) in enumerate(self.q_shapes):
            offset_q = 0
            for level_s, (hs, ws) in enumerate(self.s_shapes):
                ind = torch.arange(ws).view(1, -1) / ws - \
                        torch.arange(hq).view(-1, 1) / hq
                indx = ind.repeat(wq, hs) * math.sqrt(ws * hq)
                indy = ind.repeat_interleave(wq, dim=0).repeat_interleave(
                    hs, dim=1) * math.sqrt(ws * hq)
                indd = indx**2 + indy**2

                rel_indices[offset_s:offset_s + hq*wq, offset_q:offset_q + hs*ws,
                            0] = indd
                rel_indices[offset_s:offset_s + hq*wq, offset_q:offset_q + hs*ws,
                            1] = indx
                rel_indices[offset_s:offset_s + hq*wq, offset_q:offset_q + hs*ws,
                            2] = indy
                offset_q += hs * ws
            offset_s += hq * wq

        self.rel_indices = rel_indices.unsqueeze(0)


class QuerySupportBlock(nn.Module):
    def __init__(self,
                 cfg,
                 in_dim,
                 out_dim,
                 num_heads,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 average_support_first=True,
                 **kwargs):
        super().__init__()

        self.cfg = cfg
        self.average_support_first = average_support_first

        self.in_norm_query = norm_layer(in_dim)
        self.in_norm_support = norm_layer(in_dim)
        self.attn = QSGA(cfg,
                         in_dim,
                         out_dim,
                         self.average_support_first,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attn_drop=attn_drop,
                         proj_drop=drop,
                         **kwargs)


        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm_out = norm_layer(in_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=in_dim,
            hidden_features=mlp_hidden_dim,
            # out_features=256,
            act_layer=act_layer,
            drop=drop)

    def forward(self, q_feat, s_feat, q_shapes, s_shapes):
        # reduce dimensionality
        # q_feat = self.input_proj(q_feat)
        # s_feat = self.input_proj(s_feat)

        q_feat = self.in_norm_query(q_feat)
        s_feat = self.in_norm_support(s_feat)
        B, Nq, C = q_feat.shape
        M, Ns, C = s_feat.shape

        K = self.cfg.FEWSHOT.K_SHOT

        if self.average_support_first:
            # average support features per class
            s_feat = s_feat.reshape(M // K, K, Ns, C).mean(dim=1)
            M = M // K

        # QSGA is memory heavy, use pytorch checkpoint API to trade memory
        # with computation when K_SHOT is too large. Maybe use K_SHOT * N_WAYS
        # as a criterion instead.
        if self.cfg.FEWSHOT.K_SHOT > 300:
            attention_q_feat = torch.utils.checkpoint.checkpoint(
                self.attn, q_feat, s_feat, q_shapes, s_shapes)
        else:
            attention_q_feat = self.attn(q_feat, s_feat, q_shapes, s_shapes)


        q_feat = q_feat.unsqueeze(1).repeat(1, M, 1, 1)#.flatten(end_dim=1).unsqueeze(1)

        # Regular QSGA
        q_feat = q_feat + self.drop_path(attention_q_feat)
        q_feat = q_feat + self.drop_path(self.mlp(self.norm_out(q_feat)))

        # NO skip
        # q_feat = self.drop_path(attention_q_feat)
        # q_feat = self.drop_path(self.mlp(self.norm_out(q_feat)))

        # NO MLP
        # q_feat = self.drop_path(attention_q_feat)


        q_feat = q_feat.reshape(B,M,Nq,C)

        return q_feat
