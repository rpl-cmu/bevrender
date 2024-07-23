import torch
import numpy as np
from PIL import Image
from einops import rearrange
import torchvision.transforms.functional as F
from scipy.spatial.transform import Rotation as R

torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
np.set_printoptions(precision=2, suppress=True)


class BEV2CameraProjector:
    def __init__(
        self,
        imu_to_rgb,
        K,
        vehicle_type_code,
        img_width,
        img_height,
        ori_img_width,
        ori_img_height,
        remove_ref_in_gray=False,
        bound_check_img_paths=None,
        device="cuda",
        logger=None,
        use_wandb=False,
    ):
        self.scale_x = img_width / ori_img_width
        self.scale_y = img_height / ori_img_height
        self.img_width = img_width
        self.img_height = img_height
        self.imu_to_cmr = imu_to_rgb
        self.K = K
        self.vehicle_type_code = vehicle_type_code
        self.remove_ref_in_gray = remove_ref_in_gray
        self.bound_check_img_paths = bound_check_img_paths
        self.device = device
        self.logger = logger
        self.use_wandb = use_wandb

        for key in self.K.keys():
            for value in self.K[key]:
                value[0, 0] *= self.scale_x
                value[0, 2] *= self.scale_x
                value[1, 1] *= self.scale_y
                value[1, 2] *= self.scale_y

        self.imu_to_cmr = self.to_tensor(self.imu_to_cmr)
        self.K = self.to_tensor(self.K)

        self.img_paths = {self.vehicle_type_code: self.bound_check_img_paths}

    def to_tensor(self, input):
        output = {}
        for key in input.keys():
            output[key] = []
            for value in input[key]:
                output[key].append(torch.tensor(value).float())
        return output

    def bev_grid_to_camera(self, points_3d):
        WIDTH_IDX, HEIGHT_IDX = 0, 1
        h, w, z = points_3d.shape[1:]
        points_3d = rearrange(points_3d, "n h w z -> n (h w z)")
        points_2d_dict = {}
        imu_2_cmr = self.imu_to_cmr[self.vehicle_type_code]
        intrinsic_K = self.K[self.vehicle_type_code]
        points_2d_list = []

        for module in range(len(imu_2_cmr)):
            """TODO add a matrix - world to imu, needs to be done for each frame"""
            points_cmr_rect = imu_2_cmr[module].inverse() @ points_3d
            points_2d = intrinsic_K[module][:, :3] @ points_cmr_rect[:3]
            points_2d = points_2d.div(points_2d[-1])[:2]
            mask = self.get_in_bound_mask(points_2d, self.vehicle_type_code, module)
            points_2d = points_2d.masked_fill(~mask, 0)

            """plot 3D reference points on example image"""
            # example_image = Image.open(self.img_paths[self.vehicle_type_code][module])
            # draw = ImageDraw.Draw(example_image)
            # draw_size = 2
            # for i in range(points_2d.shape[1]):
            #     shape = (
            #         points_2d[WIDTH_IDX][i] - draw_size,
            #         points_2d[HEIGHT_IDX][i] - draw_size,
            #         points_2d[WIDTH_IDX][i] + draw_size,
            #         points_2d[HEIGHT_IDX][i] + draw_size,
            #     )
            #     draw.rectangle(shape, fill="red")
            # example_image.save("/home/lihongj/workspace/bev/tmp/test.png")

            """NOTE points_2d (width, height)"""
            # points_2d = points_2d.swapaxes(0, 1).unsqueeze(0)
            """all the coordinates are normalized to [-1, 1] for points_2d_dict"""
            points_2d[WIDTH_IDX] = points_2d[WIDTH_IDX] / (self.img_width - 1)
            points_2d[HEIGHT_IDX] = points_2d[HEIGHT_IDX] / (self.img_height - 1)
            points_2d = points_2d * 2 - 1
            points_2d_list.append(
                rearrange(points_2d, "n (h w z) -> n h w z", h=h, w=w, z=z)
            )
            points_2d_dict[self.vehicle_type_code] = points_2d_list
        """NOTE points_2d in the order of (x, y) - with bound (672,384)"""
        return points_2d_dict

    def get_in_bound_mask(self, points_2d, vehicle_type, module):
        points_2d = points_2d.to(torch.int32)
        WIDTH_IDX, HEIGHT_IDX = 0, 1
        mask = (
            points_2d[HEIGHT_IDX].ge(0)
            & points_2d[HEIGHT_IDX].lt(self.img_height - 1)
            & points_2d[WIDTH_IDX].ge(0)
            & points_2d[WIDTH_IDX].lt(self.img_width - 1)
        )
        if self.remove_ref_in_gray:
            ref_img = F.pil_to_tensor(Image.open(self.img_paths[vehicle_type][module]))
            points_2d = points_2d.masked_fill(~mask, 0)

            """values: (3, 15680)"""
            values = ref_img[:, points_2d[HEIGHT_IDX], points_2d[WIDTH_IDX]]
            new_mask = values == 128
            new_mask = new_mask.sum(axis=0) != 3
            return mask & new_mask
        else:
            return mask
