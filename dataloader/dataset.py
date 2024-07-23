import torch
import random
import numpy as np
from einops import rearrange
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = False

"""
data_list items:
    [
        0  - timestamp: float
        1  - rgb cmr full path: str
        
        #    gps info:
        2  - map img full path: str
        3  - vehicle pose utm northing: float
        4  - vehicle pose utm easting: float
        5  - vehicle pose utm height(negative): float
        6  - vehicle pose euler angle roll: float
        7  - vehicle pose euler angle pitch: float
        8  - vehicle pose euler angle yaw: float
        9  - vehicle pose pixel coordinate x: float
        10 - vehicle pose pixel coordinate y: float

        11 - vehicle type code: int
    ]
"""


class GPSDeniedDataset(Dataset):
    def __init__(
        self,
        datalist,
        mode,
        data_augmentation,
        batch_size,
        num_views,
        window_num_imgs,
        resize_cmr_img,
        resize_img_height,
        resize_img_width,
        img_norm_mean,
        img_norm_std,
        map_norm_mean,
        map_norm_std,
        logger=None,
    ):
        self.logger = logger
        self.datalist = datalist
        self.resize_image = resize_cmr_img
        self.img_resize_height = resize_img_height
        self.img_resize_width = resize_img_width
        self.mode = mode
        self.batch_size = batch_size
        self.window_num_imgs = window_num_imgs
        self.num_views = num_views

        self.map_norm_mean = map_norm_mean
        self.map_norm_std = map_norm_std

        self.img_norm_transform = transforms.Normalize(
            mean=img_norm_mean, std=img_norm_std
        )

        self.map_transform = self.get_map_transform()
        self.img_transform = self.get_img_transform(data_augmentation)

    """
        returns training dataset, validation dataset, inference dataset w.r.t. code
    """

    def __getitem__(self, index):
        """
        Function:
            get item from data_list

        Args:
            index (int): index of data_list

        Returns:
            dictionary: {
                "camera":       torch.Tensor(6,3,512,640),
                "map":          torch.Tensor(3,224,224),
                "vehicle_pose": torch.Tensor(6,3),
                "vehicle_type": torch.Tensor(1)
            }
        """

        (
            TIMESTAMP_IDX,
            RGB_IMG_IDX,
            MAP_IMG_IDX,
            VEH_POS_X_IDX,
            VEH_POS_Y_IDX,
            VEH_POS_HEADING_IDX,
            VEH_TYPE_IDX,
        ) = (0, 1, 2, 9, 10, 8, 11)

        (
            image_tensor_list,
            vehicle_pose_tensor_list,
        ) = ([], [])

        if self.mode == "train" or self.mode == "validation":
            idx = set(
                random.sample(
                    range(len(self.datalist[index][:-1])),
                    k=self.window_num_imgs,
                )
            )
            temp_data_list = [
                x for i, x in enumerate(self.datalist[index][:-1]) if i in idx
            ]
            temp_data_list.append(self.datalist[index][-1])
        elif self.mode == "inference":
            temp_data_list = self.datalist[index]

        for item in temp_data_list:
            image_tensor_list.append(
                rearrange(
                    self.img_transform(
                        Image.fromarray(
                            np.concatenate(
                                (np.array(Image.open(item[RGB_IMG_IDX])),),
                                axis=1,
                            )
                        )
                    ),
                    "c h (n w) -> n c h w",
                    n=self.num_views,
                )
                / 255.0
            )

            vehicle_pose_tensor_list.append(
                torch.tensor(
                    [
                        item[i]
                        for i in [VEH_POS_X_IDX, VEH_POS_Y_IDX, VEH_POS_HEADING_IDX]
                    ]
                )
            )
        image_tensor = torch.stack(image_tensor_list)
        image_tensor = self.img_norm_transform(image_tensor)
        vehicle_pose_tensor = torch.stack(vehicle_pose_tensor_list)

        map_img = Image.open(self.datalist[index][-1][MAP_IMG_IDX])
        map_tensor = self.map_transform(map_img)

        timestamp = int(self.datalist[index][-1][TIMESTAMP_IDX])

        vehicle_type_tensor = torch.tensor(
            self.datalist[index][-1][VEH_TYPE_IDX]
        ).unsqueeze(0)

        return {
            "timestamp": timestamp,
            "camera": image_tensor,
            "map": map_tensor,
            "vehicle_pose": vehicle_pose_tensor,
            "vehicle_type": vehicle_type_tensor,
        }

    def __len__(self):
        return len(self.datalist)

    def get_img_transform(self, augmentation_type):
        """
        perform camera image:
            1. resize
            2. data augmentation
        """
        img_transform = []

        if self.resize_image:
            img_transform.append(
                transforms.Resize((self.img_resize_height, self.img_resize_width))
            )

        img_transform.append(transforms.PILToTensor())

        if augmentation_type == "strong":
            img_transform.extend(
                [
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomPosterize(p=0.2, bits=4),
                    # transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)),
                ]
            )

        elif augmentation_type == "weak":
            img_transform.extend(
                [
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )

        elif augmentation_type == "none":
            pass

        else:
            raise RuntimeError("wrong data augmentation type!")

        return transforms.Compose(img_transform)

    def get_map_transform(self):
        """
        perform map image:
            1. scaling to 0~1
            2. normalization w.r.t. mean & std NOTE: no normalization for bev rendering
        """
        img_transform = [
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.map_norm_mean, std=self.map_norm_std),
        ]
        return transforms.Compose(img_transform)
