import torch
from torch import nn
import torch.nn.functional as F

from .. import registry
from ..utils.qsga import QuerySupportBlock
from .utils import AAFIOMode


class BaseAlignment(nn.Module):
    def __init__(self, cfg, align_first, aaf_cfg, *args):
        super().__init__(*args)

        self.cfg = cfg
        self.aaf_cfg = aaf_cfg

        # match right keys for input and output in features dict.
        self.input_name = '_features' if align_first else '_p1'
        self.output_name = '_p1' if align_first else '_p2'

        self.in_mode = AAFIOMode.Q_BCHW_S_NCHW
        self.out_mode = AAFIOMode.Q_BNCHW_S_NBCHW
    
    def forward(self, features):
        """
        Arguments regrouped in features dict with at least the following keys 
        (it can change when align_first=False):
         - query_features: extracted features from query image with backbone 
                            List[Tensor] B x Channels x H x W
         - support_features: backbone features for each of the support images
                            List[Tensor] N_support x Channels x H x W
         - support_targets: targets boxes and label corresponding to each 
                            support imageList[BoxList]
        
        Returns inside the dict:
         - support_p1: support features aligned with query
                                List[Tensor] N_support x B x Channels x H x W
         - query_p1: query features aligned with support
                                List[Tensor] B x N_support x Channels x H x W
        """
        pass


@registry.ALIGNMENT_MODULE.register
class IDENTITY(BaseAlignment):
    """
    Identity alignment with repeating the query and support
    to match the requested BNCHW and NBCHW.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.in_mode = AAFIOMode.ID
        self.out_mode = AAFIOMode.ID

    def forward(self, features):
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        N_support = support_features[0].shape[0]
        B = query_features[0].shape[0]

        if N_support % (B * self.cfg.FEWSHOT.K_SHOT) != 0 or N_support // (
                B * self.cfg.FEWSHOT.K_SHOT) == 1:

            support_aligned_query = [
                level.unsqueeze(0).repeat(B, 1, 1, 1, 1).permute(1,0,2,3,4)
                for level in support_features
            ]
        else:
            support_aligned_query = [
                level.view(-1, B, *level.shape[-3:])
                for level in support_features
            ]
            N_support = support_aligned_query[0].shape[0]

        query_aligned_support = [
            level.unsqueeze(0).repeat(N_support, 1, 1, 1, 1).permute(1,0,2,3,4)
            for level in query_features
        ]

        features.update({
            'query' + self.output_name: query_aligned_support,
            'support' + self.output_name: support_aligned_query
        })


@registry.ALIGNMENT_MODULE.register
class IDENTITY_NO_REPEAT(BaseAlignment):
    """
    Identity alignment without repeating the query and support
    to match the requested BNCHW and NBCHW
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.in_mode = AAFIOMode.ID
        self.out_mode = AAFIOMode.ID

    def forward(self, features):
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']


        features.update({
            'query' + self.output_name: query_features,
            'support' + self.output_name: support_features
        })


@registry.ALIGNMENT_MODULE.register
class SIMILARITY_ALIGN(BaseAlignment):
    """
    Similarity alignment module for Meta Faster R-CNN: Towards Accurate 
    Few-Shot Object Detection with Attentive Feature Alignment
    (https://arxiv.org/abs/2104.07719)
    """
    def __init__(self, *args):
        super().__init__(*args)


    def forward(self, features):
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        support_aligned_query = []
        query_aligned_support = []
        self.attention_map = []
        K = self.cfg.FEWSHOT.K_SHOT

        for query, support in zip(query_features, support_features):
            N_support, C, Hs, Ws = support.shape
            B, C, Hq, Wq = query.shape
            query = query.flatten(-2).permute(0, 2, 1).unsqueeze(1).repeat(
                1, N_support, 1, 1).flatten(end_dim=1) #BNxHqWqxC
            support = support.flatten(-2).unsqueeze(0).repeat(
                B, 1, 1, 1).flatten(end_dim=1)  #BNxCxHsWs

            query = query - query.mean()
            support = support - support.mean()

            sim_matrix = torch.bmm(query, support) #BNxHqWqxHsWs
            # softmax
            sim_matrix = F.softmax(sim_matrix, dim=-1)
            aligned_support = torch.bmm(sim_matrix, support.permute(0, 2, 1))
            aligned_support = aligned_support.reshape(
                B, N_support//K, K, Hq, Wq, -1).mean(dim=2).permute(1, 0, 4, 2, 3) #Ns,B,C,Hq,Wq
            query = query.permute(0, 2, 1).reshape(
                B, N_support//K, K, C, Hq, Wq).mean(dim=2) #B,Ns,C,Hq,Wq
            support_aligned_query.append(aligned_support)
            query_aligned_support.append(query)

            # self.attention_map.append(sim_matrix.detach().cpu())

        features.update({
            'query' + self.output_name: query_aligned_support,
            'support' + self.output_name: support_aligned_query,
            'attention_map': self.attention_map
        })


@registry.ALIGNMENT_MODULE.register
class CISA(BaseAlignment):
    """
    CISA block for Dual-Awareness Attention for
    Few-Shot Object Detection
    (https://arxiv.org/abs/2102.12152)
    """
    def __init__(self, *args, in_ch=256):
        super().__init__(*args)

        self.W_r = nn.Linear(in_ch, 1)
        self.W_k = nn.Linear(in_ch, in_ch // 4)
        self.W_q = nn.Linear(in_ch, in_ch // 4)

        self.in_mode = AAFIOMode.Q_BCHW_S_NCHW
        self.out_mode = AAFIOMode.Q_BNCHW_S_NBCHW

        self.query_attended_features = None

    def forward(self, features):
        query_features = features['query' + self.input_name]
        support_features = features['support' + self.input_name]
        support_targets = features['support_targets']

        support_aligned_ = []
        query_aligned_ = []
        self.attention_map = []

        K = self.cfg.FEWSHOT.K_SHOT

        for query, support in zip(query_features, support_features):

            B, C, Hq, Wq = query.shape
            # N_support should be N_WAYS*K_SHOT * B
            # but if SAME_SUPPORT_IN_BATCH is False, it could be
            # N_WAYS*K_SHOT only.
            N_support, C, Hs, Ws = support.shape

            # if N_support % (B * self.cfg.FEWSHOT.K_SHOT) != 0 or N_support // (
            #         B * self.cfg.FEWSHOT.K_SHOT) == 1:
            if (N_support // K == B) or (N_support % (B * self.cfg.FEWSHOT.K_SHOT) != 0):
                # support = support.repeat(B, 1, 1, 1)
                support = support.repeat_interleave(B, dim=0)
                N_support, C, Hs, Ws = support.shape

            query = query.repeat(N_support // B, 1, 1, 1) #N, C, Hq, Wq

            query = query.flatten(start_dim=-2).permute(0, 2, 1)  # N, HqWq, C
            support = support.flatten(start_dim=-2).permute(0, 2,1)  # N, HsWs, C
            # print(query.shape, support.shape)

            query_mapped = self.W_q(query) # N, HqWq, C'
            support_mapped = self.W_k(support).permute(0, 2, 1)  # N, C', HsWs
            query_mapped = query_mapped - query_mapped.mean()
            support_mapped = support_mapped - support_mapped.mean()

            query_support_sim = torch.bmm(query_mapped, support_mapped) # N, HqWq, HsWs
            query_support_sim = query_support_sim / Hq
            query_support_sim = F.softmax(
                query_support_sim.view(B, -1),
                dim=-1).view_as(query_support_sim)  # N, HqWq, HsWs

            support_self = F.softmax(self.W_r(support), dim=1) # N, HsWs, 1
            query_support_sim = query_support_sim + 0.1 * support_self.permute(
                0, 2, 1)  # B, HqWq, HsWs

            support_aligned = torch.bmm(query_support_sim, support) # N, HqWq, C
            support_aligned = support_aligned.view(
                N_support//B//K, K, B, Hq, Wq, C).mean(dim=1).permute(0,1,4,2,3) # N_WAYS*K_SHOT, B, C, Hq, Wq

            query_aligned = query.view(
                N_support // B // K, K, B, Hq, Wq, C).mean(dim=1).permute(1, 0, 4, 2, 3)

            support_aligned_.append(support_aligned)
            query_aligned_.append(query_aligned)

        self.query_attended_features = support_aligned_
        features.update({
            'query' + self.output_name: query_aligned_,
            'support' + self.output_name: support_aligned_
        })


@registry.ALIGNMENT_MODULE.register
class XQSA(BaseAlignment):
    """
    Identity alignment without repeating the query and support
    to match the requested BNCHW and NBCHW
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.in_mode = AAFIOMode.Q_BCHW_S_NCHW
        self.out_mode = AAFIOMode.Q_BNCHW_S_NBCHW

        self.num_head = self.aaf_cfg.ALIGNMENT.NUM_HEADS

        self.average_support_first = self.aaf_cfg.ALIGNMENT.AVERAGE_SUPPORT_FIRST
        self.qsga_block = QuerySupportBlock(
            self.cfg,
            self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
            self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
            self.num_head,
            average_support_first=self.average_support_first)


    def forward(self, features):
        K = self.cfg.FEWSHOT.K_SHOT
        q_feat, q_shapes, s_feat, s_shapes = self.reshape_input_features(
            features)
        support_targets = features['support_targets']

        q_shapes = torch.tensor(q_shapes)
        s_shapes = torch.tensor(s_shapes)

        aligned_feat = self.qsga_block(q_feat, s_feat, q_shapes, s_shapes)
        # aligned_feat = torch.utils.checkpoint.checkpoint(self.qsga_block, q_feat, s_feat, q_shapes, s_shapes)
        aligned_feat = aligned_feat.permute(0,1,3,2)
        out_features_list = []

        B, N_ways, C, _ = aligned_feat.shape
        offset = 0
        for (h,w) in q_shapes:
            if self.average_support_first:
                out_features_list.append(aligned_feat[:, :, :, offset:offset +
                                                      h * w].reshape(
                                                          B, N_ways, C, h, w))
            else:
                out_features_list.append(
                    aligned_feat[:,:,:,offset:offset + h * w].reshape(B, N_ways // K, K, C, h, w).mean(dim=2))

            # already averaged

            offset += h * w
        support_features_aligned = [support.unsqueeze(1).repeat(1, B, 1, 1, 1)
                                    for support in features['support' + self.input_name]]
        features.update({
            'query' + self.output_name: out_features_list,
            'support' + self.output_name: support_features_aligned
        })

    def reshape_input_features(self, features):
        q_feat = features['query' + self.input_name]
        s_feat = features['support' + self.input_name]

        q_shapes = [feat.shape[-2:] for feat in q_feat]
        s_shapes = [feat.shape[-2:] for feat in s_feat]

        q_feat = torch.cat(
            [feat.flatten(start_dim=-2).permute(0, 2, 1) for feat in q_feat],
            dim=1)
        s_feat = torch.cat(
            [feat.flatten(start_dim=-2).permute(0, 2, 1) for feat in s_feat],
            dim=1)
        return q_feat, q_shapes, s_feat, s_shapes


    # """
    # Cross-Scales Query-Support Alignment (XQSA) for enhancing small object detection in FSOD.
    # align_first = True: Align support features to query features before the attention mechanism.
    # """
    # def __init__(self, cfg, align_first=True, *args, **kwargs):
    #     super().__init__(cfg, align_first=align_first, *args, **kwargs)
    #     self.align_first = align_first
    #     self.d = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # Dimensionality of feature vectors
    #     self.WQ = nn.Linear(self.d, self.d)  # Transformation for query
    #     self.WK = nn.Linear(self.d, self.d)  # Transformation for keys
    #     self.WV = nn.Linear(self.d, self.d)  # Transformation for values
    #     self.mlp = nn.Sequential(
    #         nn.LayerNorm(self.d),
    #         nn.Linear(self.d, self.d),
    #         nn.ReLU(),
    #         nn.Linear(self.d, self.d)
    #     )
    
    # def forward(self, features):
    #     B, C, _, _ = features['query' + self.input_name][0].shape  # Example shape from the first FPN level
    #     N, _, _, _ = features['support' + self.input_name][0].shape  # Dimensions of support features

    #     # Flatten features from all scales and concatenate
    #     query_flat = torch.cat([f.view(B, C, -1) for f in features['query' + self.input_name]], dim=2)  # [B, C, H*W*levels]
    #     support_flat = torch.cat([f.view(N, C, -1) for f in features['support' + self.input_name]], dim=2)  # [N, C, H'*W'*levels]

    #     # Project to query, key, and value spaces
    #     Q = self.WQ(query_flat.permute(0, 2, 1))  # [B, H*W*levels, C]
    #     K = self.WK(support_flat.permute(0, 2, 1))  # [N, H'*W'*levels, C]
    #     V = self.WV(support_flat.permute(0, 2, 1))  # [N, H'*W'*levels, C]

    #     # Prepare Q, K, V for batch processing
    #     Q = Q.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, H*W*levels, C]
    #     K = K.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, H'*W'*levels, C]
    #     V = V.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, H'*W'*levels, C]

    #     # Compute affinity matrix for each query-support pair
    #     affinity = torch.einsum('bnqc,bnkc->bnqk', Q, K)  # [B, N, H*W*levels, H'*W'*levels]
    #     lambda_c = F.softmax(affinity / torch.sqrt(torch.tensor(self.d, dtype=torch.float32)), dim=-1)  # Softmax over keys

    #     # Apply attention to values
    #     A_c_q = torch.einsum('bnqk,bnkc->bnqc', lambda_c, V)  # [B, N, H*W*levels, C]
        

    #     # Apply MLP with skip connection (expand and reshape appropriately)
    #     A_c_q_transformed = self.mlp(A_c_q)
    #     A_c_q = A_c_q + A_c_q_transformed  # Element-wise addition for skip connection

    #     # Reshape back to original spatial dimensions for each FPN level
    #     query_aligned_support = []
    #     start = 0
    #     for level_features in features['query' + self.input_name]:
    #         _, _, H, W = level_features.shape
    #         num_elements = H * W
    #         end = start + num_elements
    #         reshaped = A_c_q[:, :, start:end].view(B, N, C, H, W)
    #         query_aligned_support.append(reshaped)
    #         start = end

    #     features['query' + self.output_name] = query_aligned_support  # List([B, N, C, H, W]) ready for further processing
        
    #     support_features = features['support' + self.input_name]

    #     if N % (B * self.cfg.FEWSHOT.K_SHOT) != 0 or N // (
    #             B * self.cfg.FEWSHOT.K_SHOT) == 1:

    #         support_aligned_query = [
    #             level.unsqueeze(0).repeat(B, 1, 1, 1, 1).permute(1,0,2,3,4)
    #             for level in support_features
    #         ]
    #     else:
    #         support_aligned_query = [
    #             level.view(-1, B, *level.shape[-3:])
    #             for level in support_features
    #         ]
    #         N = support_aligned_query[0].shape[0]

    #     features['support' + self.output_name] = support_aligned_query

    #     # return features
    