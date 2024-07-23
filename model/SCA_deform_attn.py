import sys
import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

sys.path.append(str(Path.cwd()))

from model.model_utils import LayerNormProxy


class SCADeformableAttention(nn.Module):
    def __init__(
        self,
        bev_feat_shape,
        bev_depth_dim,
        dim_embed,
        n_heads,
        n_groups,
        stride,
        kernel_size,
        scale_offset_range,
        batch_size,
        n_views=3,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        data_type=torch.float32,
        logger=None,
    ):
        super().__init__()
        self.n_channel_per_head = dim_embed // n_heads  # 32
        self.scale = self.n_channel_per_head**-0.5
        self.n_heads = n_heads  # 2
        self.embed_dim = self.n_channel_per_head * n_heads  # 32*2=64
        self.n_groups = n_groups  # 1
        self.n_channel_per_group = self.embed_dim // n_groups  # 64
        self.n_heads_per_group = self.n_heads // n_groups  # 2
        self.query_height, self.query_width = bev_feat_shape, bev_feat_shape  # 64
        self.bev_depth_dim = bev_depth_dim
        self.batch_size = batch_size
        self.scale_offset_range = scale_offset_range

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_views = n_views
        self.data_type = data_type
        self.logger = logger
        pad_size = kernel_size // 2 if kernel_size != stride else 0

        """offset_range_factor: 0.5 -> 0.9%, 1.0 -> 0.18%, 5.0 -> 9%"""
        if self.scale_offset_range:
            self.offset_range_factor = 5.0

        self.conv_offset_m0 = nn.Sequential(
            nn.Conv2d(
                self.n_channel_per_group,
                self.n_channel_per_group * self.bev_depth_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.n_channel_per_group,
                # nn.Tanh(),
                # Scale(offset_scale),
            ),
            LayerNormProxy(self.n_channel_per_group * self.bev_depth_dim),
            nn.GELU(),
            nn.Conv2d(
                self.n_channel_per_group * self.bev_depth_dim,
                self.bev_depth_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.conv_offset_m1 = nn.Sequential(
            nn.Conv2d(
                self.n_channel_per_group,
                self.n_channel_per_group * self.bev_depth_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.n_channel_per_group,
                # nn.Tanh(),
                # Scale(offset_scale),
            ),
            LayerNormProxy(self.n_channel_per_group * self.bev_depth_dim),
            nn.GELU(),
            nn.Conv2d(
                self.n_channel_per_group * self.bev_depth_dim,
                2 * self.bev_depth_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.conv_offset_m2 = nn.Sequential(
            nn.Conv2d(
                self.n_channel_per_group,
                self.n_channel_per_group * self.bev_depth_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.n_channel_per_group,
                # nn.Tanh(),
                # Scale(offset_scale),
            ),
            LayerNormProxy(self.n_channel_per_group * self.bev_depth_dim),
            nn.GELU(),
            nn.Conv2d(
                self.n_channel_per_group * self.bev_depth_dim,
                2 * self.bev_depth_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.proj_q = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_k = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_v = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = nn.Conv2d(
            self.embed_dim * self.n_views,
            self.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_views = nn.Conv2d(
            self.n_channel_per_group * self.n_views,
            self.n_channel_per_group,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.proj_drop_rate = nn.Dropout(proj_drop_rate, inplace=True)
        self.attn_drop_rate = nn.Dropout(attn_drop_rate, inplace=True)

        self.rpe_table = nn.Parameter(
            torch.zeros(
                self.n_heads,
                self.query_height * 2 - 1,
                self.query_width * self.bev_depth_dim * 2 - 1,
            )
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_normalized_grid(self, H, W, B, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=self.data_type, device=device),
            torch.arange(0, W, dtype=self.data_type, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(
        self,
        x,
        query,
        reference_points,
        wandb_log_dict,
        return_wandb_log=True,
    ):
        """
        Function:
            forward pass of SpatialDeformableAttention

        Args:
            x:                  (bs * n_views, embed_dim, img_h, img_w)     -> (2,  3, 64,  56, 56)
            query:              (bs, embed_dim, bev_h, bev_w)               -> (2, 64, 56,  56)
            reference_points:   (bs * n_views, ref_h, ref_w * ref_d, 2)     -> (2,  3, 56, 280,  2)

        Returns:
            out:        (bs, embed_dim, bev_h, bev_w)                       -> (2, 64,  56, 56)
        """

        device = query.device

        """change to (y, x) order to align with grid_sample & offset"""
        reference_points = reference_points[..., (1, 0)]
        reference_points = repeat(
            reference_points, "b v h w n -> (b g) v h w n", g=self.n_groups
        )

        Hq, Wq = query.shape[-2:]
        B, _, C, Hi, Wi = x.size()  # (2, 3, 64, 56, 56)
        output_list = []

        for view_idx in range(self.n_views):
            x_view = x[:, view_idx, ...]
            ref_view = reference_points[:, view_idx, ...]

            """reshaped_q: (bs * n_groups, embed_dim, bev_h, bev_w) -> (bs * g, 64 / g, 56, 56))"""
            """offset: (bs * n_groups, n_views * bev_d, bev_h, bev_w) -> (2, 5, 56, 56)"""
            if view_idx == 0:
                offset = self.conv_offset_m0(
                    rearrange(
                        query,
                        "b (g c) h w -> (b g) c h w",
                        g=self.n_groups,
                        c=self.n_channel_per_group,
                    ).contiguous()
                )
            elif view_idx == 1:
                offset = self.conv_offset_m1(
                    rearrange(
                        query,
                        "b (g c) h w -> (b g) c h w",
                        g=self.n_groups,
                        c=self.n_channel_per_group,
                    ).contiguous()
                )
            elif view_idx == 2:
                offset = self.conv_offset_m2(
                    rearrange(
                        query,
                        "b (g c) h w -> (b g) c h w",
                        g=self.n_groups,
                        c=self.n_channel_per_group,
                    ).contiguous()
                )
            """offset: (bs * n_groups, n_views * bev_d, bev_h, bev_w) -> (2, 5, 56, 56)"""
            """offset: (bs * n_groups, 2, bev_h, bev_w * bev_d) -> (2, 2, 28, 280)"""
            offset = rearrange(
                offset,
                "(b g) d (h n) w -> (b g) n h (w d)",
                b=self.batch_size,
                g=self.n_groups,
                d=self.bev_depth_dim,
                n=2,
            ).contiguous()

            Hk, Wk = offset.shape[-2:]
            n_sample = Hk * Wk  # 7480 = 28 * 280

            """limit offset to a certain range - offset_range_factor: 0.5 -> 8%, 1.0 -> 16%"""
            if self.scale_offset_range:
                offset_range = (
                    torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device)
                    .contiguous()
                    .reshape(1, 2, 1, 1)
                )
                offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

            """offset: (bs * n_groups, bev_h, bev_w * bev_d, 2) -> (2, 28, 280, 2)"""
            offset = rearrange(offset, "b n h w -> b h w n", n=2).contiguous()

            if self.scale_offset_range:
                ref_w_offset = offset + ref_view
            else:
                ref_w_offset = (offset + ref_view).clamp(
                    -1.0, +1.0
                )  # clamp to keep in range -1.0 ~ +1.0

            """
            Function:
                samples kv feature from x based on pos(offset + reference)
            
            Args:
                input:      (bs * n_groups, embed_dim, bev_h, bev_w)          -> (2, 64, 128, 160)
                grid:       (bs * n_groups, bev_h, bev_w * bev_d, 2)          -> (2, 28, 280,   2)
            
            Return:
                x_sampled:  (bs * n_groups, embed_dim, bev_h, bev_w * bev_d)  -> (2, 64,  28, 280)
            """
            x_sampled = (
                F.grid_sample(
                    input=x_view.contiguous().reshape(
                        B * self.n_groups, self.n_channel_per_group, Hi, Wi
                    ),
                    grid=ref_w_offset[..., (1, 0)],  # TODO double check: y, x -> x, y
                    mode="bilinear",
                    align_corners=True,
                )
                .contiguous()
                .reshape(B, C, 1, n_sample)
            )

            """q: (bs * n_heads, n_channel_per_head, bev_h * bev_w) -> (4, 32, 3136) NOTE n_heads=2"""
            q = query.contiguous().reshape(
                B * self.n_heads, self.n_channel_per_head, Hq * Wq
            )

            """
            k:  (bs * n_heads, n_channel_per_head, bev_h * bev_w * bev_d)   -> (4, 32, 7840) NOTE n_heads=2
            v:  (bs * n_heads, n_channel_per_head, bev_h * bev_w * bev_d)   -> (4, 32, 7840) NOTE n_heads=2
            """
            k = (
                self.proj_k(x_sampled)
                .contiguous()
                .reshape(B * self.n_heads, self.n_channel_per_head, n_sample)
            )
            v = (
                self.proj_v(x_sampled)
                .contiguous()
                .reshape(B * self.n_heads, self.n_channel_per_head, n_sample)
            )

            """
            input of multi-head_attn:
                q: (4, 32,  3136)
                k: (4, 32, 7840)
                v: (4, 32, 7840)
            output:
                attn: (bs * n_heads, bev_h * bev_w, bev_h * bev_w * bev_d) -> (4, 3136, 7840)
            """
            attn = torch.einsum("b c m, b c n -> b m n", q, k).contiguous()
            attn = attn.mul(self.scale)  # attn shape (16, 3136, 15680)
            # if return_wandb_log:
            #     wandb_log_dict["sca_attn"] = wandb.Image(
            #         Image.fromarray(attn[0].clone().detach().cpu().numpy()).convert(
            #             "L"
            #         )
            #     )

            """rpe_table: (n_heads, bev_h * 2 - 1, bev_w * bev_d * 2 - 1) -> (2, 111, 559)"""
            rpe_table = self.rpe_table

            """rpe_bias: (bs * n_groups, n_heads, bev_h * 2 - 1, bev_w * 2 - 1) -> (2, 2, 111, 111)"""
            rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

            """
            NOTE
                1) q_grid: (bs * n_groups, bev_h, bev_w, 2) -> (2, 56, 56, 2)
                2) q_grid is query grid, since relative position bias encodes relative position between query and key
                3) order of q_grid is (y, x) / (h, w)
            """
            q_grid = self._get_normalized_grid(
                Hq, Wq, B, device
            )  # pass in (56, 56, 2), output range: -1.0 ~ +1.0

            """
            reshape:
                q_grid:         (bs * n_groups, 56,  56, 2) -> (bs * num_groups, 3136,    1, 2)   -> (2, 3136,  1, 2)
                ref_w_offset:   (bs * n_groups, 28, 280, 2) -> (bs * num_groups,    1, 7840, 2)   -> (2,    1, 49, 2)
                
            output:
                displacement:   (bs * n_groups, bev_h * bev_w, bev_h * bev_w * bev_d, 2)    -> (2, 3136, 7840, 2)
                NOTE multiply by 0.5 to make displacement range: -1.0 ~ +1.0
            """
            displacement = (
                q_grid.contiguous().reshape(B * self.n_groups, Hq * Wq, 2).unsqueeze(2)
                - ref_w_offset.contiguous()
                .reshape(B * self.n_groups, n_sample, 2)
                .unsqueeze(1)
            ).mul(0.5)

            """
            input:
                rpe_bias:   (bs * n_groups, n_heads_per_group, bev_h * 2 - 1, bev_w * bev_d * 2 - 1) -> (2, 2, 111, 559)
                reshape rpt_bias:   (2, 2, 111, 559)
            output:
                attn_bias:  (bs * n_groups, n_heads, bev_h * bev_w, bev_h * bev_w * bev_d) -> (2, 2, 3136, 7840)
            """
            attn_bias = F.grid_sample(
                input=rearrange(
                    rpe_bias.contiguous(),
                    "b (g n) h w -> (b g) n h w",
                    n=self.n_heads_per_group,
                    g=self.n_groups,
                ).contiguous(),
                grid=displacement[..., (1, 0)],
                mode="bilinear",
                align_corners=True,
            )  # (2, 2, 3136, 15680)

            """attn_bias:   (bs * n_heads, bev_h * bev_w, bev_h * bev_w * bev_d) -> (4, 3136, 7840)"""
            attn_bias = attn_bias.contiguous().reshape(
                B * self.n_heads, Hq * Wq, n_sample
            )

            """attn:        (bs * n_heads, bev_h * bev_w, bev_h * bev_w * bev_d) -> (4, 3136, 7840)"""
            attn = attn + attn_bias

            # """numerical stability from lucidrains"""
            # attn = attn - attn.amax(dim=-1, keepdim=True).detach()

            attn = F.softmax(attn, dim=2)
            # if return_wandb_log:
            #     wandb_log_dict["sca_attn_bias"] = wandb.Image(
            #         Image.fromarray(attn[0].clone().detach().cpu().numpy()).convert(
            #             "L"
            #         )
            #     )
            attn = self.attn_drop_rate(attn)

            out = torch.einsum("b m n, b c n -> b c m", attn, v).contiguous()
            out = out.reshape(B, C, Hq, Wq)
            output_list.append(out)

        output = torch.stack(output_list, dim=1)
        output = rearrange(
            output, "b v c h w -> b (v c) h w", v=self.n_views
        ).contiguous()

        output = self.proj_drop_rate(self.proj_out(output))
        return output, wandb_log_dict
