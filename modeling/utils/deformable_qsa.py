import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from .convit import Mlp


class DeformableQSA(nn.Module):
    def __init__(self,
                 cfg,
                 in_dim,
                 out_dim,
                 n_pts=3,
                 num_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 locality_strength=1.,):
        super().__init__()

        self.cfg = cfg

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_pts = n_pts
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.k = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, out_dim, bias=qkv_bias)

        self.delta = nn.Sequential(nn.Linear(2 * in_dim, 2 * n_pts * num_heads),
                                   nn.Sigmoid()
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)



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

        q = self.q(q_feat).reshape(B, Nq, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(s_feat).reshape(M, Ns, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(s_feat).reshape(M, Ns, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)

        q_feat_repeated = q_feat.unsqueeze(1).expand(B, M, Nq, C)
        s_feat_repeated = s_feat.unsqueeze(0).expand(B, M, Ns, C)
        s_vect = s_feat_repeated.mean(dim=-2, keepdim=True).expand_as(q_feat_repeated)
        q_feat_s_feat = torch.cat([q_feat_repeated, s_vect], dim=-1)

        delta = self.delta(q_feat_s_feat).reshape(B, M, Nq, self.num_heads, 
                                   self.n_pts, 2).permute(0, 1, 3, 2, 4, 5)
        
        q_shapes_sq = q_shapes ** 2
        s_shapes_sq = s_shapes ** 2
        shape_per_loc = torch.tensor(s_shapes, device=s_feat.device).repeat_interleave(q_shapes_sq.to(s_feat.device))
        shape_per_loc_sq = torch.tensor(s_shapes_sq, device=s_feat.device).repeat_interleave(q_shapes_sq.to(s_feat.device))
        shape_per_loc = shape_per_loc.reshape(1, 1, 1, -1, 1)
        shape_per_loc_sq = shape_per_loc_sq.reshape(1, 1, 1, -1, 1)

        # features are flatten row first --> convert shifts into flattened positions
        delta = delta[...,1] * shape_per_loc + delta[...,0] * shape_per_loc_sq 
        delta = delta.long()
        delta = delta.flatten(start_dim=3, end_dim=4).unsqueeze(-1)
        delta = delta.expand(*delta.shape[:-1], v.shape[-1])
        
        q = q.unsqueeze(1).expand(B, M, self.num_heads, Nq, C // self.num_heads)
        k = k.unsqueeze(0).expand(B, M, self.num_heads, Ns, C // self.num_heads)
        v = v.unsqueeze(0).expand(B, M, self.num_heads, Ns, C // self.num_heads)  #BM, N_h, Ns, C // N_h

        k = k.gather(-2, delta).reshape(
            B, M, self.num_heads, Nq, self.n_pts, C//self.num_heads)
        v = v.gather(-2, delta).reshape(
            B, M, self.num_heads, Nq, self.n_pts, C//self.num_heads)
        

        attn = torch.einsum('bmhqc,bmhqpc->bmhqp', q, k)
        attn = torch.softmax(attn, dim=-1)

        query_aligned_features = torch.einsum('bmhqp,bmhqpc->bmhqc', attn, v)  #BM, Nh, Nq, C//Nh

        query_aligned_features = query_aligned_features.transpose(
            2, 3).flatten(start_dim=-2)
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


        # patch_score = torch.einsum('bhqc,mhsc->bmhqs', q,
        #                            k).flatten(end_dim=1) * self.scale

        q = q.unsqueeze(1).repeat(1, M, 1, 1, 1)
        k = k.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        patch_score = q@k.permute(0,1,2,4,3)
        patch_score = patch_score.flatten(end_dim=1) * self.scale

        patch_score = patch_score.softmax(dim=-1)

        pos_score = pos_score.softmax(dim=-1)  #BM, N_q, N_s, 3

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)
                ) * patch_score + torch.sigmoid(gating) * pos_score

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
        attn_map = self.get_attention(q_feat, s_feat).reshape(B, M // K, K, -1, Nq, Ns)
        attn_map = attn_map.mean(2)  # average over support
        distances = self.rel_indices.squeeze()[:, :, -1]**.5
        dist = torch.einsum('nm,bwhnm->bwh', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist




class DeformableQuerySupportBlock(nn.Module):
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
                 **kwargs):
        super().__init__()

        self.cfg = cfg

        self.in_norm_query = norm_layer(in_dim)
        self.in_norm_support = norm_layer(in_dim)
        self.attn = DeformableQSA(cfg,
                         in_dim,
                         out_dim,
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
        M = s_feat.shape[0]

        # QSGA is memory heavy, use pytorch checkpoint API to trade memory
        # with computation when K_SHOT is too large. Maybe use K_SHOT * N_WAYS
        # as a criterion instead.
        if self.cfg.FEWSHOT.K_SHOT > 3:
            attention_q_feat = torch.utils.checkpoint.checkpoint(
                self.attn, q_feat, s_feat, q_shapes, s_shapes)
        else:
            attention_q_feat = self.attn(q_feat, s_feat, q_shapes, s_shapes)


        q_feat = q_feat.unsqueeze(1).repeat(1, M, 1, 1)#.flatten(end_dim=1).unsqueeze(1)
        q_feat = q_feat + self.drop_path(attention_q_feat)
        # q_feat = self.drop_path(attention_q_feat)
        # print("q_feat mean: {}, attention_q_feat mean: {}".format(q_feat.mean(), attention_q_feat.mean()))
        # print("q_feat std: {}, attention_q_feat std: {}".format(q_feat.std(), attention_q_feat.std()))
        # q_feat = torch.cat([q_feat.unsqueeze(1).repeat(1, M, 1, 1),


        q_feat = q_feat + self.drop_path(self.mlp(self.norm_out(q_feat)))
        # q_feat = self.mlp(self.norm_out(q_feat))

        q_feat = q_feat.reshape(B,M,Nq,C)

        return q_feat
