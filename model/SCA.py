import torch
import torch.nn as nn
from einops import rearrange, repeat

from model.SCA_deform_attn import SCADeformableAttention


class SpatialCrossAttn(nn.Module):
    def __init__(
        self,
        bev_bound,
        bev2cmr_projector,
        bev_feat_shape,
        bev_depth_dim,
        z_shift,
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
        self.bev_bound = bev_bound
        self.bev_feat_shape = bev_feat_shape
        self.bev_depth_dim = bev_depth_dim
        self.z_shift = z_shift
        self.logger = logger
        self.batch_size = batch_size
        self.num_views = n_views
        """self.points_2d_dict range: [-1,1]"""
        self.points_2d_dict = bev2cmr_projector.bev_grid_to_camera(
            self.sample_3d_points()
        )
        assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"
        assert n_heads // n_groups >= 1, "n_heads must be greater than n_groups"
        self.spatial_deform_attn = SCADeformableAttention(
            bev_feat_shape=bev_feat_shape,
            bev_depth_dim=bev_depth_dim,
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

    def forward(
        self, query, img_feat, vehicle_type_idx, wandb_log_dict, return_wandb_log=True
    ):
        """
        Function:
            forward pass of SpatialCrossAttn

        Args:
            query:                4D    - (bs, bev_embed_dim, bev_h, bev_w)
            img_feat:             5D    - (bs, num_views, model_dim, img_feat_h, img_feat_w)
            vehicle_type_idx:     int   - 0, 1, 2, 3, 4, 5, 6

        Returns:
            output:               4D - (bs, bev_embed_dim, bev_h, bev_w)
        """
        device = query.device

        """reference_points: (n_views, 2, bev_h, bev_w, bev_d) -> (1, 2, 28, 56, 5)"""
        reference_points = torch.stack(
            self.points_2d_dict[vehicle_type_idx.item()], dim=0
        ).to(device=device)

        """reference_points: (bs, n_views, bev_h, bev_w * bev_d, n) -> (2, 3, 28, 56*5, 2)"""
        reference_points = repeat(
            reference_points, "v n h w d -> b v h (w d) n", b=self.batch_size
        )

        """img_feat: (bs, n_views, model_dim, img_feat_h, img_feat_w) -> (2, 3, 64, 128, 160)"""
        if len(img_feat.shape) == 4:
            img_feat = rearrange(
                img_feat,
                "(b v) d h w -> b v d h w",
                b=self.batch_size,
                v=self.num_views,
            ).contiguous()

        """
        input:
            reference_points: (bs * n_views, bev_d * bev_h, bev_w, 2) -> (2,  3, 28, 280,  2)
            img_feat:         (bs * n_views, model_dim, img_h, img_w) -> (2,  3, 64,  56, 56)
        output:
            cur_bev_feat:     (bs, bev_embed_dim, bev_h, bev_w) -> (6,64,56,56)
        """
        cur_bev_feat, wandb_log_dict = self.spatial_deform_attn(
            x=img_feat,
            query=query,
            reference_points=reference_points,
            wandb_log_dict=wandb_log_dict,
            return_wandb_log=return_wandb_log,
        )
        return cur_bev_feat, wandb_log_dict

    def sample_3d_points(self):
        """
        Function:
            sample 3d points in bev space w.r.t. bev x, y, z bound and bev_feat_shape

        Return:
            points_3d in homogenous coordinates
            points_3d_homo:                         (4, bev_h * bev_w * bev_z)
        """
        x_shift = self.bev_bound["X"] / self.bev_feat_shape
        x_resolution = x_shift * 2
        y_shift = self.bev_bound["Y"] / self.bev_feat_shape
        y_resolution = y_shift * 2
        z_shift = self.bev_bound["Z"] / self.bev_depth_dim
        z_resolution = z_shift * 2

        grid = torch.stack(
            torch.meshgrid(
                torch.arange(
                    0 + x_shift,
                    self.bev_bound["X"] + x_shift,
                    x_resolution,
                ),
                torch.arange(
                    -self.bev_bound["Y"] + y_shift,
                    self.bev_bound["Y"] + y_shift,
                    y_resolution,
                ),
                indexing="ij",
            ),
            dim=0,
        )
        sample_height = torch.arange(
            -self.bev_bound["Z"] + z_shift + self.z_shift,
            self.bev_bound["Z"] + z_shift + self.z_shift,
            z_resolution,
        )
        sample_height = sample_height.repeat(
            grid.shape[-2], grid.shape[-1], 1
        ).unsqueeze(0)
        grid = grid.repeat(sample_height.shape[-1], 1, 1, 1)
        grid = rearrange(grid, "n ... -> ... n")

        ones = torch.ones_like(sample_height)
        points_3d_homo = torch.cat((grid, sample_height, ones), dim=0)

        """voxel_grid: ( 4([x,y,z,1] -- homogeneous coord), bev_height * bev_width * bev_depth )"""
        # points_3d_homo = rearrange(points_3d_homo, "n h w z -> n (h w z)")
        points_3d_homo.requires_grad = False
        """points_3d in homogenous coordinates, shape: (4, bev_h, bev_w, bev_z)"""
        return points_3d_homo
