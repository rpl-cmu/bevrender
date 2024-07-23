import sys
import torch
import torch.nn as nn
from pathlib import Path
from einops import rearrange, repeat

sys.path.append(str(Path.cwd()))

from model.encoder import BEVEncoder
from model.decoder_img_render import BEVImageRenderDecoder
from model.bev_cmr_proj import BEV2CameraProjector


class BEVRender(nn.Module):
    def __init__(self, config, logger, mode):
        super().__init__()
        self.logger = logger

        self.batch_size = config["BATCH_SIZE"] if mode == "train" else 1
        self.data_type = config["DATA_TYPE"]

        self.init_bev_height = config["DAT_BEV_SHAPE"][0]
        self.init_bev_width = config["DAT_BEV_SHAPE"][0]
        self.init_embed_dim = config["DAT_EMBED_DIMS"][0]

        self.last_bev_height = config["DAT_BEV_SHAPE"][-1]
        self.last_bev_width = config["DAT_BEV_SHAPE"][-1]
        self.last_embed_dim = config["DAT_EMBED_DIMS"][-1]

        bev2cam_projector = BEV2CameraProjector(
            vehicle_type_code=config["VEHICLE_TYPE_CODE"],
            imu_to_rgb=config["IMU_TO_RGB"],
            K=config["INTRINSIC_K"],
            img_height=config["IMG_HEIGHT"],
            img_width=config["IMG_WIDTH"],
            ori_img_height=config["ORI_IMG_HEIGHT"],
            ori_img_width=config["ORI_IMG_WIDTH"],
            remove_ref_in_gray=config["REMOVE_REF_IN_GRAY"],
            bound_check_img_paths=config["BOUND_CHECK_IMG_PATH"],
            logger=logger,
        )

        self.encoder = BEVEncoder(
            bev_bound=config["BEV_BOUND"],
            bev2cmr_projector=bev2cam_projector,
            batch_size=self.batch_size,
            scale_offset_range=config["DAT_SCALE_OFFSET_RANGE"],
            n_stages=config["DAT_NUM_STAGES"],
            n_views=config["NUM_VIEWS"],
            expansion=config["DAT_EXPANSION"],
            dims=config["DAT_EMBED_DIMS"],
            bev_feat_shapes=config["DAT_BEV_SHAPE"],
            bev_depth_dim=config["DAT_BEV_DEPTH_DIM"],
            z_shift=config["SAMPLE_Z_SHIFT"],
            depths=config["DAT_VIT_DEPTHS"],
            n_heads=config["DAT_NUM_HEADS"],
            strides=config["DAT_STRIDES"],
            n_groups=config["DAT_NUM_GROUPS"],
            kernel_size=config["DAT_K_SIZES"],
            drop_rate=config["DAT_DROP_RATE"],
            attn_drop_rate=config["DAT_ATTN_DROP_RATE"],
            drop_path_rate=config["DAT_DROP_PATH_RATE"],
            backbone_arch=config["DAT_BACKBONE_TYPE"],
            data_type=config["DATA_TYPE"],
            logger=logger,
        )

        """set up decoder"""
        self.decoder = BEVImageRenderDecoder(
            bev_spatial_dim=config["DAT_BEV_SHAPE"][-1],
            model_dim=config["DAT_EMBED_DIMS"][-1],
            hid_dim=config["DECODER_HID_DIM"],
            logger=logger,
        )

        """self.bev_temporal_window = BEVWindowQueue(
            num_of_samples_in_window=config["UNORDERED_WINDOW_LENGTH"],
            window_time_spin=config["UNORDERED_WINDOW_TIMESPIN"] * 1e6,
        )"""

        self.bev_embedding = nn.Embedding(
            self.init_bev_height * self.init_bev_width,
            self.init_embed_dim,
        )

        self.init_weights()

    def forward(
        self,
        img_tensor,
        vehicle_pose_tensor,
        vehicle_type_tensor,
        wandb_log_dict,
        return_wandb_log=True,
    ):
        """
        Function:
            forward pass of BEVFormer

            1. prev_bev_embed = get_history_bev(prev_imgs, vehicle_poses, vehicle_type): require_grid=False
            2. bev_query = bev_query + positional_embedding
            3. bev_query = encoder(bev_query, cur_img_feat, prev_bev_embed, vehicle_motion, vehicle_type_idx)
            4. output = decoder(bev_query)

        Args:
            img_tensor:             (bs, num_samples_in_window, num_views, 3, 224, 224) -> (2,4,3,3,224,224)
            vehicle_pose_tensor:    (bs, num_samples_in_window, 3)                      -> (2,4,3)
            vehicle_type_tensor:    (bs, 1)                                             -> (2,1)

        Returns:
            output:                 (bs, channel, img_height, img_width)                -> (2,3,224,224)
        """

        """set up bev_query data, repeat and reshape to 4D"""
        bev_query = self.bev_embedding.weight.to(self.data_type).to(img_tensor.device)
        bev_query = repeat(bev_query, "n d -> b n d", b=self.batch_size)
        bev_query = rearrange(
            bev_query,
            "b (h w) d -> b d h w",
            h=self.init_bev_height,
            w=self.init_bev_width,
        )
        vehicle_type_idx = vehicle_type_tensor[0, 0]
        self.eval()
        with torch.no_grad():
            prev_bev, wandb_log_dict = self.get_history_bev(
                bev_query,
                img_tensor[:, :-1, ...],
                vehicle_pose_tensor,
                vehicle_type_idx,
                wandb_log_dict=wandb_log_dict,
                return_wandb_log=False,
            )
        self.train()
        assert prev_bev.shape == bev_query.shape

        """shape of bev_query remain the same - (batch_size, bev_embed_dim, bev_h, bev_w) -> (2, 64, 56, 56)"""
        bev_query = self.encoder(
            bev_query=bev_query,
            img_tensor=img_tensor[:, -1, ...],
            prev_bev=prev_bev,
            vehicle_pose=vehicle_pose_tensor[:, -1, ...],
            vehicle_type_idx=vehicle_type_idx,
            wandb_log_dict=wandb_log_dict,
            return_wandb_log=return_wandb_log,
        )
        assert bev_query.shape == prev_bev.shape

        output = self.decoder(bev_query)
        return output, wandb_log_dict

    def init_weights(self):
        for m in self.modules():
            # if isinstance(m, ResnetFPN):
            #     self.init_resnetfpn(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight.data)
        # self.logger.info(f"BEVFormer weights initialized!")

    def init_resnetfpn(self, m):
        """TODO initialize weights of ResNet for ResnetFPN"""
        pass

    def get_history_bev(
        self,
        bev_query,
        img_tensor,
        vehicle_pose,
        vehicle_type_idx,
        wandb_log_dict,
        return_wandb_log=False,
    ):
        """
        Function:
            get prev_bev from t-3 to t-1 img_tensor and bev_query

        Args:
            bev_query:              (bs, bev_embed_dim, bev_h, bev_w)                       -> (2,64,56,56)
            img_tensor:             (bs, num_samples_in_window-1, num_views, 3, 224, 224)   -> (2,3,3,3,224,224)
            vehicle_pose:           (bs, num_samples_in_window, 3)                          -> (2,4,3)
            vehicle_type_idx:       int

        Returns:
            prev_bev:               (bs, bev_embed_dim, bev_h, bev_w)                       -> (2,64,56,56)
        """
        prev_bev = None
        assert img_tensor.shape[1] == vehicle_pose.shape[1] - 1
        for i in range(img_tensor.shape[1]):
            """
            bev_query:          (2,256,56,56)
            img_tensor:         (2,3,3,224,224)
            prev_bev:           (2,256,56,56) or None
            vehicle_pose:       (2,3)
            vehicle_type_idx:   int
            """
            prev_bev = self.encoder(
                bev_query,
                img_tensor[:, i, ...],
                prev_bev,
                vehicle_pose[:, i : i + 2, ...],
                vehicle_type_idx,
                wandb_log_dict=wandb_log_dict,
                return_wandb_log=return_wandb_log,
            )
            assert prev_bev.shape == bev_query.shape
        return prev_bev, wandb_log_dict
