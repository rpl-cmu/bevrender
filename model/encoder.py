import math
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from model.SCA import SpatialCrossAttn
from model.TSA import TemporalSelfAttn
from model.feedforward import FeedForwardLayer
from model.img_backbone import PatchProjection, ResNet18_wo_fpn
from model.model_utils import TransformerMLPWithConv, LayerNormProxy


class BEVEncoder(nn.Module):
    def __init__(
        self,
        bev_bound,
        bev2cmr_projector,
        batch_size,
        scale_offset_range,
        n_stages=7,
        n_views=3,
        expansion=4,
        dims=[64, 128, 256, 512, 256, 128, 64, 64],
        bev_feat_shapes=[56, 28, 14, 7, 14, 28, 56, 56],
        bev_depth_dim=5,
        z_shift=-1.0,
        depths=[2, 2, 2, 2, 2, 2, 2],
        n_heads=[2, 4, 8, 16, 8, 4, 2],
        strides=[8, 4, 2, 1, 2, 4, 8],
        n_groups=[1, 2, 4, 8, 4, 2, 1],
        kernel_size=[9, 7, 5, 3, 5, 7, 9],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        backbone_arch="ResNet18",
        data_type=torch.float32,
        logger=None,
    ):
        super().__init__()
        self.logger = logger
        # self.img_backbone = ResnetFPN(backbone_arch, logger=logger)

        if backbone_arch == "ResNet18":
            self.img_backbone = ResNet18_wo_fpn(
                bev_dim=bev_feat_shapes[0], logger=logger
            )
        elif backbone_arch == "PatchProjection":
            if bev_feat_shapes[0] == 56:
                self.img_backbone = PatchProjection(dims[0], 4, logger=logger)
            elif bev_feat_shapes[0] == 28:
                self.img_backbone = PatchProjection(dims[0], 8, logger=logger)
            elif bev_feat_shapes[0] == 14:
                self.img_backbone = PatchProjection(dims[0], 16, logger=logger)

        self.stages = nn.ModuleList([])

        for stage_idx in range(n_stages):
            self.stages.append(
                BEVEncoderStage(
                    bev_bound=bev_bound,
                    bev2cmr_projector=bev2cmr_projector,
                    batch_size=batch_size,
                    scale_offset_range=scale_offset_range,
                    stage_idx=stage_idx,
                    n_views=n_views,
                    expansion=expansion,
                    dims=dims[stage_idx : stage_idx + 2],  # [64]
                    bev_feat_shapes=bev_feat_shapes[stage_idx : stage_idx + 2],  # [56]
                    bev_depth_dim=bev_depth_dim,
                    z_shift=z_shift,
                    depth=depths[stage_idx],
                    n_heads=n_heads[stage_idx],
                    strides=strides[stage_idx],
                    n_groups=n_groups[stage_idx],
                    kernel_size=kernel_size[stage_idx],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    data_type=data_type,
                    logger=logger,
                )
            )

    def forward(
        self,
        bev_query,
        img_tensor,
        prev_bev,
        vehicle_pose,
        vehicle_type_idx,
        wandb_log_dict,
        return_wandb_log=True,
    ):
        """take view to batch dimension"""
        if len(img_tensor.shape) == 5:
            img_tensor = rearrange(img_tensor, "b v c h w -> (b v) c h w")

        """
        fpn_img_feat - tuple of 4 tensors, shape:
        0 - (batch_size * n_views,  64, 56, 56)
        1 - (batch_size * n_views, 128, 28, 28)
        2 - (batch_size * n_views, 256, 14, 14)
        3 - (batch_size * n_views, 512,  7,  7)
        etc.
        """
        """img_tensor: (bs * num_views, img_feat_num_channels, img_feat_h, img_feat_w) --> (8, 64, 128, 160)"""
        img_tensor = self.img_backbone(img_tensor)  # min max: (-1.8,1.7)
        for stage_idx, stage in enumerate(self.stages):
            if prev_bev is not None:
                assert (
                    bev_query.shape[1:]
                    == prev_bev.shape[1:]
                    # == img_tensor[stage_idx if stage_idx < 4 else 6 - stage_idx].shape[1:] # for ResNet18_FPN
                ), f"bev_query.shape: {bev_query.shape}, prev_bev.shape: {prev_bev.shape}"
            bev_query, wandb_log_dict = stage(
                bev_query=bev_query,
                # img_tensor=img_tensor[stage_idx if stage_idx < 4 else 6 - stage_idx], # for ResNet18_FPN
                img_tensor=img_tensor,  # for patch projection
                prev_bev=prev_bev,
                vehicle_pose=vehicle_pose,
                vehicle_type_idx=vehicle_type_idx,
                wandb_log_dict=wandb_log_dict,
                return_wandb_log=return_wandb_log,
            )
        return bev_query


class BEVEncoderStage(nn.Module):
    def __init__(
        self,
        bev_bound,
        bev2cmr_projector,
        batch_size,
        scale_offset_range,
        stage_idx=0,
        n_views=3,
        expansion=4,
        dims=[64, 128],
        bev_feat_shapes=[56, 28],
        bev_depth_dim=5,
        z_shift=-1.0,
        depth=2,
        n_heads=2,
        strides=8,
        n_groups=1,
        kernel_size=9,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        data_type=torch.float32,
        logger=None,
    ):
        super().__init__()
        self.logger = logger

        """self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem),
        )"""

        self.curr_feat_dim, self.next_feat_dim = (
            dims if len(dims) == 2 else (dims[0], dims[0])
        )
        self.curr_bev_feat_shape, self.next_bev_feat_shape = (
            bev_feat_shapes
            if len(bev_feat_shapes) == 2
            else (bev_feat_shapes[0], bev_feat_shapes[0])
        )
        self.encoder_layers = nn.ModuleList([])

        if self.curr_bev_feat_shape == self.next_bev_feat_shape:
            # self.stage_project_conv = nn.Conv2d(
            #     self.curr_feat_dim, self.next_feat_dim, 1, 1, 0
            # )
            self.stage_project_conv = nn.Identity()
        elif self.curr_bev_feat_shape > self.next_bev_feat_shape:
            self.stage_project_conv = nn.Conv2d(
                self.curr_feat_dim, self.next_feat_dim, 3, 2, 1
            )
        else:
            self.stage_project_conv = nn.ConvTranspose2d(
                self.curr_feat_dim, self.next_feat_dim, kernel_size=2, stride=2
            )

        for _ in range(depth):
            self.encoder_layers.append(
                EncoderLayer(
                    bev_bound=bev_bound,
                    bev2cmr_projector=bev2cmr_projector,
                    n_views=n_views,
                    bev_feat_shape=self.curr_bev_feat_shape,
                    bev_depth_dim=bev_depth_dim,
                    z_shift=z_shift,
                    dim_embed=self.curr_feat_dim,
                    expansion=expansion,
                    stage_idx=stage_idx,
                    n_groups=n_groups,
                    n_heads=n_heads,
                    stride=strides,
                    kernel_size=kernel_size,
                    batch_size=batch_size,
                    scale_offset_range=scale_offset_range,
                    attn_drop_rate=attn_drop_rate,
                    proj_drop_rate=drop_rate,
                    mlp_drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    data_type=data_type,
                    logger=logger,
                )
            )

    def forward(
        self,
        bev_query,
        img_tensor,
        prev_bev,
        vehicle_pose,
        vehicle_type_idx,
        wandb_log_dict,
        return_wandb_log=True,
    ):
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            assert bev_query.shape == prev_bev.shape if prev_bev is not None else True
            bev_query, wandb_log_dict = encoder_layer(
                bev_query=bev_query,
                img_tensor=img_tensor,
                prev_bev=prev_bev,
                vehicle_pose=vehicle_pose,
                vehicle_type_idx=vehicle_type_idx,
                wandb_log_dict=wandb_log_dict,
                return_wandb_log=return_wandb_log,
            )
        bev_query = self.stage_project_conv(bev_query)
        return bev_query, wandb_log_dict


class EncoderLayer(nn.Module):
    def __init__(
        self,
        bev_bound,
        bev2cmr_projector,
        n_views,
        bev_feat_shape,
        bev_depth_dim,
        z_shift,
        dim_embed,
        expansion,
        stage_idx,
        n_groups,
        n_heads,
        stride,
        kernel_size,
        batch_size,
        scale_offset_range,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_drop_rate=0.0,
        drop_path_rate=0.2,
        ffn_drop_rate=0.1,
        data_type=torch.float32,
        logger=None,
    ):
        super().__init__()
        self.logger = logger
        self.stage_idx = stage_idx
        self.bev_feat_shape = bev_feat_shape

        self.layer_scale = nn.Identity()
        self.layer_norm = LayerNormProxy(dim_embed)
        self.tsa_mlp = TransformerMLPWithConv(dim_embed, expansion, mlp_drop_rate)
        self.sca_mlp = TransformerMLPWithConv(dim_embed, expansion, mlp_drop_rate)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.tsa_local_percept_unit = nn.Conv2d(
            dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed
        )
        self.sca_local_percept_unit = nn.Conv2d(
            dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed
        )
        self.down_proj = nn.Sequential(
            nn.Conv2d(dim_embed, dim_embed * 2, 3, 2, 1, bias=False),
            LayerNormProxy(dim_embed * 2),
        )
        self.ffn_tsa = FeedForwardLayer(
            in_dim=bev_feat_shape, hidden_dim=dim_embed, dropout=ffn_drop_rate
        )
        self.ffn_sca = FeedForwardLayer(
            in_dim=bev_feat_shape, hidden_dim=dim_embed, dropout=ffn_drop_rate
        )

        self.temporal_self_attn = TemporalSelfAttn(
            bev_feat_shape=bev_feat_shape,
            dim_embed=dim_embed,
            n_heads=n_heads,
            n_groups=n_groups,
            stride=stride,
            kernel_size=kernel_size,
            batch_size=batch_size,
            scale_offset_range=scale_offset_range,
            n_views=n_views,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            data_type=data_type,
            logger=logger,
        )
        self.spatial_cross_attn = SpatialCrossAttn(
            bev_bound=bev_bound,
            bev2cmr_projector=bev2cmr_projector,
            bev_feat_shape=bev_feat_shape,
            bev_depth_dim=bev_depth_dim,
            z_shift=z_shift,
            dim_embed=dim_embed,
            n_heads=n_heads,
            n_groups=n_groups,
            stride=stride,
            kernel_size=kernel_size,
            batch_size=batch_size,
            scale_offset_range=scale_offset_range,
            n_views=n_views,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            data_type=data_type,
            logger=logger,
        )
        """self.ffn = FeedForwardLayer(
            in_dim=bev_embed_dim, hidden_dim=ffn_hid_dim, dropout=ffn_dropout
        )
        self.norm1 = nn.LayerNorm(bev_embed_dim)
        self.norm2 = nn.LayerNorm(bev_embed_dim)
        self.norm3 = nn.LayerNorm(bev_embed_dim)"""

    def forward(
        self,
        bev_query,
        img_tensor,
        prev_bev,
        vehicle_pose,
        vehicle_type_idx,
        wandb_log_dict,
        return_wandb_log=True,
    ):
        """
        Function:
            forward pass of one encoder layer

        Args:
            bev_query:              (bs, bev_embed_dim, bev_h, bev_w)                                   -> (2, 64, 56, 56)
            img_tensor:             (bs * num_views, img_feat_num_channels, img_feat_h, img_feat_w)     -> (6, 64, 128, 160)
            prev_bev:               (bs, bev_embed_dim, bev_h, bev_w)                                   -> (2, 64, 56, 56)        or None
            vehicle_pose:           (bs, 2, 3)                                                          -> (2,  2,  3)
            vehicle_type_idx:       int

        Returns:
            output:                 (bs, bev_embed_dim, bev_h, bev_w)                                   -> (2, 256, 56, 56)
        """
        x = bev_query

        """perform temporal self attention"""
        if prev_bev is not None and not self.training:
            prev_bev = self.project_history_bev_feat(prev_bev, vehicle_pose)

        """perform temporal self attention"""
        x = x + self.tsa_local_percept_unit(x.contiguous())
        x0 = x
        x, wandb_log_dict = self.temporal_self_attn(
            query=self.layer_norm(x).contiguous(),
            prev_bev=prev_bev,
            wandb_log_dict=wandb_log_dict,
            return_wandb_log=return_wandb_log,
        )
        x = self.layer_scale(x)
        x = self.drop_path(x) + x0
        x0 = x

        """NOTE should we add a FFN here?"""
        # x = self.ffn_tsa(x)

        x = self.tsa_mlp(self.layer_norm(x).contiguous())
        x = self.layer_scale(x)
        x = self.drop_path(x) + x0

        """perform spatial cross attention"""
        x = x + self.sca_local_percept_unit(x.contiguous())
        x0 = x
        x, wandb_log_dict = self.spatial_cross_attn(
            query=self.layer_norm(x).contiguous(),
            img_feat=img_tensor,
            vehicle_type_idx=vehicle_type_idx,
            wandb_log_dict=wandb_log_dict,
            return_wandb_log=return_wandb_log,
        )
        x = self.layer_scale(x)
        x = self.drop_path(x) + x0
        x0 = x

        """NOTE should we add a FFN here?"""
        # x = self.ffn_sca(x)

        x = self.sca_mlp(self.layer_norm(x).contiguous())
        x = self.layer_scale(x)
        x = self.drop_path(x) + x0

        """x: (bs, bev_embed_dim, bev_h, bev_w)"""
        return x, wandb_log_dict

    def project_history_bev_feat(self, bev, vehicle_pose, return_mask=False):
        """
        Function:
            project history bev feature to current bev frame

        Args:
            bev: history bev feature (bs, embed_dim, bev_h, bev_w)
            vehicle_motion: vehicle motion (bs, 2, 3)

        Returns:
            projected_bev: (bs, embed_dim, bev_h, bev_w)
            bev_mask: (bs, bev_h, bev_w)
        """
        ROTATION_IDX = 2
        projected_bev_list = []
        if return_mask:
            bev_mask_list = []

        for i in range(bev.shape[0]):
            prev_rot, curr_rot = vehicle_pose[i, :, ROTATION_IDX]
            delta_x, delta_y, _ = vehicle_pose[i, 0] - vehicle_pose[i, 1]

            """post-rotation translation"""
            projected_bev = F.affine(
                img=bev[i],
                angle=math.degrees(prev_rot),
                translate=(delta_x, delta_y),
                scale=1.0,
                shear=0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            projected_bev = F.affine(
                img=projected_bev,
                angle=math.degrees(-curr_rot),
                translate=(0, 0),
                scale=1.0,
                shear=0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            projected_bev_list.append(projected_bev)
            if return_mask:
                bev_mask_list.append(projected_bev != torch.zeros([3, 1, 1]))

        projected_bev = torch.stack(projected_bev_list, dim=0)
        if return_mask:
            bev_mask = torch.stack(bev_mask_list, dim=0)

        if return_mask:
            return projected_bev, bev_mask
        else:
            return projected_bev
