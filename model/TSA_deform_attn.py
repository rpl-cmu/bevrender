import sys
import torch
from torch import nn
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

sys.path.append(str(Path.cwd()))

from model.model_utils import LayerNormProxy


class TSADeformableAttention(nn.Module):
    def __init__(
        self,
        bev_feat_shape,
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
        self.bev_h, self.bev_w = bev_feat_shape, bev_feat_shape  # 64
        self.batch_size = batch_size
        self.scale_offset_range = scale_offset_range

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_views = n_views
        self.data_type = data_type
        self.logger = logger
        pad_size = kernel_size // 2 if kernel_size != stride else 0

        """offset_range_factor: 0.5 -> 8%, 1.0 -> 16%"""
        if self.scale_offset_range:
            self.offset_range_factor = 0.5

        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.n_channel_per_group,
                self.n_channel_per_group,
                kernel_size,
                stride,
                pad_size,
                groups=self.n_channel_per_group,
                # nn.Tanh(),
                # Scale(offset_scale),
            ),
            LayerNormProxy(self.n_channel_per_group),
            nn.GELU(),
            nn.Conv2d(self.n_channel_per_group, 2, 1, 1, 0, bias=False),
        )

        self.proj_q = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_k = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_v = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0
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
            torch.zeros(self.n_heads, self.bev_h * 2 - 1, self.bev_w * 2 - 1)
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_normalized_grid(self, H, W, B, device):  # same as _get_q_grid in DAT
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

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, device):  # original implementation
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H_key - 0.5, H_key, dtype=self.data_type, device=device
            ),
            torch.linspace(
                0.5, W_key - 0.5, W_key, dtype=self.data_type, device=device
            ),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(self, x, query, wandb_log_dict, return_wandb_log=True):
        """
        Function:
            forward pass of TemporalDeformableAttention

        Args:
            x:      (bs, model_dim, bev_h, bev_w)   -> (2, 64, 56, 56)
            query:  (bs, model_dim, bev_h, bev_w)   -> (2, 64, 56, 56)

        Returns:
            out:    (bs, model_dim, bev_h, bev_w)   -> (2, 64, 56, 56)
        """

        """if x is None -> no previous bev, TSA degenerate to self-attention"""
        if x is None:
            x = query.clone()

        device = x.device

        B, C, H, W = x.size()  # (2, 64, 56, 56)

        """
        1) grouped query: (bs * num_groups, num_channels / num_groups, bev_h, bev_w) -> (2, 64, 56, 56)
        2) offset spatial dimension shrink by 4 given - kernel_size=9, stride=8, padding=4, then:
           offset: (bs * n_groups, 2, bev_h, bev_w) -> (2, 2, 7, 7)
        3) keep spatial dimension of bev query as 56, we use kernel_size=3, stride=1, padding=1, thus:
           offset: (bs * n_groups, 2, bev_h, bev_w) -> (2, 2, 56, 56)
        """
        # if dist.get_rank() == 0:
        #     print("query.is_contiguous()", query.is_contiguous())
        offset = self.conv_offset(
            rearrange(
                query,
                "b (g c) h w -> (b g) c h w",
                g=self.n_groups,
                c=self.n_channel_per_group,
            ).contiguous()  # grouped query: (bs * num_groups, num_channels / num_groups, bev_h, bev_w) -> (2, 64, 56, 56)
        )
        Hk, Wk = offset.shape[-2:]
        n_sample = Hk * Wk  # 3136 = 56 * 56

        """limit offset to a certain range - offset_range_factor: 0.5 -> 8%, 1.0 -> 16%"""
        if self.scale_offset_range:
            offset_range = torch.tensor(
                [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device
            ).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        """
        offset:         (bs * n_groups, bev_h, bev_w) -> (2, 56, 56, 2)
        reference:      (bs * n_groups, bev_h, bev_w) -> (2, 56, 56, 2)
        ref_w_offset:   (bs * n_groups, bev_h, bev_w) -> (2, 56, 56, 2)
        """
        offset = rearrange(offset, "b p h w -> b h w p").contiguous()

        """
        value range for _get_normalized_grid: -1.0 ~ +1.0
        value range for _get_ref_points: -0.8333 ~ +1.1667
        NOTE using _get_normalized_grid seems to be correct
        """
        reference = self._get_normalized_grid(
            Hk, Wk, B, device
        )  # pass in (56,56,2), output shape(2,56,56,2), output range: -1.0 ~ +1.0
        if self.scale_offset_range:
            ref_w_offset = offset + reference
        else:
            ref_w_offset = (offset + reference).clamp(
                -1.0, +1.0
            )  # clamp to keep in range -1.0 ~ +1.0

        """
        Function:
            samples kv feature from x based on ref_w_offset(offset + reference)
        
        Args:
            input:      (bs * n_groups, embed_dim,  bev_h,  bev_w)      -> (2,64,56,56)
            grid:       (bs * n_groups, bev_h,      bev_w,  2)          -> (2,56,56, 2)
        
        Return:
            x_sampled:              (bs * n_groups, embed_dim, bev_h, bev_w)        -> (2,64,56,56)
            reshaped_x_sampled:     (bs * n_groups, embed_dim, 1, bev_h * bev_w)    -> (2,64,1,3136)
        """
        x_sampled = F.grid_sample(
            input=x.contiguous().reshape(
                B * self.n_groups, self.n_channel_per_group, H, W
            ),
            grid=ref_w_offset[..., (1, 0)],  # y, x -> x, y
            mode="bilinear",
            align_corners=True,
        ).reshape(B, C, 1, n_sample)

        """q: (bs * n_heads, n_channel_per_head, bev_h * bev_w) -> (4, 32, 3136) NOTE n_heads=2"""
        q = query.contiguous().reshape(B * self.n_heads, self.n_channel_per_head, H * W)

        """
        k:  (bs * n_heads, n_channel_per_head, bev_h * bev_w) -> (4, 32, 3136) NOTE n_heads=2
        v:  (bs * n_heads, n_channel_per_head, bev_h * bev_w) -> (4, 32, 3136) NOTE n_heads=2
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
            q: (4, 32, 3136)
            k: (4, 32, 3136)
            v: (4, 32, 3136)
        output:
            attn: (bs * n_heads, bev_h * bev_w, bev_h * bev_w) -> (4, 3136, 3136)
        """
        attn = torch.einsum(
            "b c m, b c n -> b m n", q, k
        ).contiguous()  # B * h, HW, Ns, attn shape for bs=2 -> (4, 3136, 3136), attn value range (-38, +32)
        attn = attn.mul(
            self.scale
        )  # attn shape for bs=2 -> (4, 3136, 3136), attn value range (-6, 6)
        """TODO remove this, attention visualization should happen after F.softmax"""
        """if return_wandb_log:
            wandb_log_dict["tsa_attn"] = wandb.Image(
                Image.fromarray(attn[0].clone().detach().cpu().numpy()).convert("L")
            )"""

        """rpe_table: (n_heads, bev_h * 2 - 1, bev_w * 2 - 1) -> (2, 111, 111)"""
        rpe_table = self.rpe_table

        """rpe_bias: (bs * num_group, n_heads, bev_h * 2 - 1, bev_w * 2 - 1) -> (2, 2, 111, 111)"""
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)  # duplicate by batch size

        """q_grid: (bs * n_groups, bev_h, bev_w, 2) -> (2, 56, 56, 2)"""
        q_grid = self._get_normalized_grid(
            H, W, B, device
        )  # pass in (56, 56, 2), output range: -1.0 ~ +1.0

        """
        reshape:
            q_grid:         (bs * num_groups, 56, 56, 2) -> (bs * num_groups, 3136, 1, 2)   -> (2, 3136, 1, 2)
            ref_w_offset:   (bs * num_groups, 56, 56, 2)   -> (bs * num_groups, 1, 49, 2)   -> (2, 1, 3136, 2)
            
        output:
            displacement: (bs * n_groups, bev_h * bev_w, bev_h * bev_w, 2) -> (2, 3136, 3136, 2)
            NOTE multiply by 0.5 to make displacement range: -1.0 ~ +1.0   
        """
        """NOTE having more than 1 group will cause trouble, should repeat q_grid before subtracting"""
        if self.n_groups > 1:
            q_grid = q_grid.repeat(self.n_groups, 1, 1, 1)
        displacement = (
            q_grid.contiguous().reshape(B * self.n_groups, H * W, 2).unsqueeze(2)
            - ref_w_offset.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
        ).mul(0.5)

        """
        input:
            rpe_bias:   (bs * num_group, num_heads_per_group, bev_h * 2 - 1, bev_w * 2 - 1) -> (2, 2, 111, 111)
            reshape rpt_bias:   (2, 2, 111, 111)
        output:
            attn_bias:  (bs * n_groups, n_heads_per_group, bev_h * bev_w, bev_h * bev_w) -> (2, 2, 3136, 3136)
        """
        attn_bias = F.grid_sample(
            input=rearrange(
                rpe_bias,
                "b (g n) h w -> (b g) n h w",
                n=self.n_heads_per_group,
                g=self.n_groups,
            ).contiguous(),
            grid=displacement[..., (1, 0)],
            mode="bilinear",
            align_corners=True,
        )  # (2, 2, 3136, 3136)

        """attn_bias:   (bs * n_heads, bev_h * bev_w, bev_h* bev_w) -> (4, 3136, 3136)"""
        attn_bias = attn_bias.contiguous().reshape(B * self.n_heads, H * W, n_sample)

        """attn:        (bs * n_heads, bev_h * bev_w, bev_h * bev_w) -> (4, 3136, 3136)"""
        attn = attn + attn_bias

        # """numerical stability from lucidrains"""
        # attn = attn - attn.amax(dim=-1, keepdim=True).detach()

        attn = F.softmax(
            attn, dim=2
        )  # attn shape for bs=2 -> (4, 3136, 3136), attn value range (0, 1)
        """TODO attention visualization happens after F.softmax"""
        # if return_wandb_log:
        #     wandb_log_dict["tsa_attn"] = wandb.Image(
        #         Image.fromarray(
        #             np.uint8((attn[0] * 255.0).clone().detach().cpu().numpy())
        #         )
        #     )
        attn = self.attn_drop_rate(attn)

        """
        attn:   (bs * n_heads, bev_h * bev_w, bev_h * bev_w) -> (4, 3136, 3136)
        v:      (bs * n_heads, n_channel_per_head, bev_h * bev_w) -> (4, 32, 3136)
        out:    (bs * n_heads, n_channel_per_head, bev_h * bev_w) -> (4, 32, 3136)
        """
        out = torch.einsum("b m n, b c n -> b c m", attn, v).contiguous()

        """out: (bs, embed_dim, bev_h, bev_w) -> (2, 64, 56, 56)"""
        out = out.reshape(B, C, H, W)

        """out: (bs, embed_dim, bev_h, bev_w) -> (2, 64, 56, 56)"""
        out = self.proj_drop_rate(self.proj_out(out))
        return out, wandb_log_dict
