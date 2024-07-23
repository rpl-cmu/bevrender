import torch
import torch.nn as nn

from model.TSA_deform_attn import TSADeformableAttention


class TemporalSelfAttn(nn.Module):
    def __init__(
        self,
        bev_feat_shape,
        dim_embed,
        n_heads,
        n_groups,
        stride,
        kernel_size,
        batch_size,
        scale_offset_range,
        n_views=3,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        data_type=torch.float32,
        logger=None,
    ):
        super().__init__()
        self.logger = logger

        assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"
        assert n_heads // n_groups >= 1, "n_heads must be greater than n_groups"

        self.temporal_deform_attn = TSADeformableAttention(
            bev_feat_shape=bev_feat_shape,
            dim_embed=dim_embed,
            n_heads=n_heads,
            n_groups=n_groups,
            stride=stride,
            kernel_size=kernel_size,
            scale_offset_range=scale_offset_range,
            batch_size=batch_size,
            n_views=n_views,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            data_type=data_type,
            logger=logger,
        )

    def forward(self, query, prev_bev, wandb_log_dict, return_wandb_log=True):
        """query, prev_bev : (bs, bev_embed_dim, bev_feat_height, bev_feat_width)"""
        """for training"""
        cur_bev_feat, wandb_log_dict = self.temporal_deform_attn(
            x=prev_bev,
            query=query,
            wandb_log_dict=wandb_log_dict,
            return_wandb_log=return_wandb_log,
        )
        return cur_bev_feat, wandb_log_dict
