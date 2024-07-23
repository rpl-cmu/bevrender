import sys
import random
import numpy as np
from pathlib import Path

sys.path.append(str(Path.cwd()))

from dataloader.dataset import GPSDeniedDataset


class DatasetProcessor:
    (
        TIMESTAMP_IDX,
        VEHICLE_TYPE_IDX,
        UTM_EASTING_IDX,
        UTM_NORTHING_IDX,
        UTM_HEIGHT_IDX,
        UTM_ROLL_IDX,
        UTM_PITCH_IDX,
        UTM_YAW_IDX,
    ) = (0, 1, 2, 3, 4, 5, 6, 7)
    split_timespin = 1e6

    def __init__(
        self,
        dataset_dir,
        overlap,
        distributed,
        k_fold,
        window_timespin,
        window_num_imgs,
        batch_size,
        num_views,
        num_workers,
        pin_memory,
        resize_cmr_img,
        resize_img_height,
        resize_img_width,
        img_norm_mean,
        img_norm_std,
        map_norm_mean,
        map_norm_std,
        gps_file_path,
        rgb_img_dir,
        map_img_dir,
        map_width,
        map_height,
        map_resize_scale,
        jgw_info,
        logger=None,
    ):
        self.logger = logger
        self.overlap = overlap
        self.distributed = distributed

        self.dataset_dir = dataset_dir
        self.gps_file_path = gps_file_path
        self.rgb_img_dir = rgb_img_dir
        self.map_img_dir = map_img_dir

        self.k_fold = k_fold
        self.window_timespin = window_timespin
        self.window_num_imgs = window_num_imgs
        self.jgw_info = jgw_info
        self.map_width = map_width
        self.map_height = map_height
        self.map_resize_scale = map_resize_scale

        self.batch_size = batch_size
        self.num_views = num_views
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.resize_cmr_img = resize_cmr_img
        self.resize_img_height = resize_img_height
        self.resize_img_width = resize_img_width
        self.img_norm_mean = img_norm_mean
        self.img_norm_std = img_norm_std
        self.map_norm_mean = map_norm_mean
        self.map_norm_std = map_norm_std

    def process_dataset(self):
        full_datalist = self.get_full_datalist()
        data_sequence_list = self.split_sequence(full_datalist)
        train_val_dataset = self.get_train_val_dataset(data_sequence_list)
        self.logger.info("processing datalist finished!")
        self.logger.info(
            "overlapping: {}, dataset size: {}\n".format(
                self.overlap, len(train_val_dataset)
            )
        )
        return train_val_dataset

    def get_train_val_dataset(self, sequence_list):
        if self.overlap:
            train_val_datalist = self.get_overlap_train_datalist(
                sequence_list=sequence_list,
                timespin=self.window_timespin,
                length=self.window_num_imgs,
            )
        else:
            train_val_datalist = self.get_train_datalist(
                sequence_list=sequence_list,
                timespin=self.window_timespin,
                length=self.window_num_imgs,
            )
        # self.logger.info(f"train_val_datalist: {len(train_val_datalist)}")

        return GPSDeniedDataset(
            train_val_datalist,
            mode="train",
            data_augmentation="none",
            batch_size=self.batch_size,
            num_views=self.num_views,
            window_num_imgs=self.window_num_imgs,
            resize_cmr_img=self.resize_cmr_img,
            resize_img_height=self.resize_img_height,
            resize_img_width=self.resize_img_width,
            img_norm_mean=self.img_norm_mean,
            img_norm_std=self.img_norm_std,
            map_norm_mean=self.map_norm_mean,
            map_norm_std=self.map_norm_std,
            logger=self.logger,
        )

    def get_train_datalist(self, sequence_list, timespin, length):
        """self.logger.info("processing non-overlapping train datalist...")"""

        train_datalist = []
        total_train_datalist_frames = 0
        TIMESTAMP_IDX = 0

        for sequence in sequence_list:
            frame_idx = 0
            while frame_idx + 1 < len(sequence):
                start_ts = sequence[frame_idx][TIMESTAMP_IDX]
                curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                candicate = []
                while curr_ts - start_ts <= timespin and frame_idx + 1 < len(sequence):
                    candicate.append(sequence[frame_idx])
                    frame_idx += 1
                    curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                if len(candicate) > length:
                    train_datalist.append(candicate)

        for train_item in train_datalist:
            total_train_datalist_frames += len(train_item)
        """self.logger.info(
            "sample size of train_datalist: {}, total frames in train_datalist: {}\n".format(
                len(train_datalist), total_train_datalist_frames
            )
        )"""
        return train_datalist

    def get_val_datalist(self, sequence_list, timespin, length, percentage):
        """self.logger.info("processing non-overlapping validation datalist...")"""

        val_datalist, candidate_datalist = [], []
        total_frames, total_val_datalist_frames, total_sequence_frames = 0, 0, 0
        TIMESTAMP_IDX = 0

        """calculate number of frames in sequecnce_list"""
        for sequence in sequence_list:
            total_frames += len(sequence)

        """get a list of candidate frames to choose from for validation given timespin and length"""
        for sequence in sequence_list:
            frame_idx = 0
            while frame_idx + 1 < len(sequence):
                start_ts = sequence[frame_idx][TIMESTAMP_IDX]
                curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                candicate = []
                while curr_ts - start_ts <= timespin and frame_idx + 1 < len(sequence):
                    candicate.append(sequence[frame_idx])
                    frame_idx += 1
                    curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                if len(candicate) > length:
                    candidate_datalist.append(candicate)

        total_candidates = len(candidate_datalist)
        total_candidate_frames = 0
        for candicate in candidate_datalist:
            total_candidate_frames += len(candicate)
        """self.logger.info(
            "total_candidate_frames: {}, total_frames: {}, percentage: {}".format(
                total_candidate_frames,
                total_frames,
                total_candidate_frames / total_frames,
            )
        )"""

        """randomly sample inf_sequences from candidate_datalist"""
        sample_idx = random.sample(
            range(total_candidates), int(total_candidates * percentage)
        )
        sample_idx.sort()
        # self.logger.info(
        #     "sample size: {}, sample_idx:\n{}\n".format(len(sample_idx), sample_idx)
        # )

        for idx in sample_idx:
            val_datalist.append(candidate_datalist[idx])

        """
        update sequence_list by removing frames in val_datalist
        to avoid duplicate frames in train_datalist and val_datalist
        """
        for val_item in val_datalist:
            for val_frame in val_item:
                for sequence in sequence_list:
                    if val_frame in sequence:
                        sequence.remove(val_frame)

        """check datalist size"""
        for val_item in val_datalist:
            total_val_datalist_frames += len(val_item)
        for sequence in sequence_list:
            total_sequence_frames += len(sequence)
        """self.logger.info(
            "val_datalist size: {}, sequence_list size: {}, original sequence_list size : {}, matched: {}\n".format(
                total_val_datalist_frames,
                total_sequence_frames,
                total_frames,
                total_val_datalist_frames + total_sequence_frames == total_frames,
            )
        )"""
        assert total_val_datalist_frames + total_sequence_frames == total_frames
        return val_datalist, sequence_list

    def get_overlap_train_datalist(self, sequence_list, timespin, length):
        """self.logger.info("processing overlapping train datalist...")"""

        train_datalist = []
        TIMESTAMP_IDX = 0

        for sequence in sequence_list:
            for frame_idx in range(len(sequence) - length):
                start_ts = sequence[frame_idx][TIMESTAMP_IDX]
                curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                candicate = []
                while curr_ts - start_ts <= timespin and frame_idx + 1 < len(sequence):
                    candicate.append(sequence[frame_idx])
                    frame_idx += 1
                    curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                if len(candicate) > length:
                    train_datalist.append(candicate)

        """self.logger.info(
            "sample size of train_datalist: {}\n".format(len(train_datalist))
        )"""
        return train_datalist

    def get_overlap_val_datalist(self, sequence_list, timespin, length, percentage):
        """self.logger.info("processing overlapping validation datalist...")"""

        val_datalist, candidate_datalist = [], []
        total_frames, total_val_datalist_frames, total_sequence_frames = 0, 0, 0
        TIMESTAMP_IDX = 0

        """calculate number of frames in sequecnce_list"""
        for sequence in sequence_list:
            total_frames += len(sequence)

        """get a list of candidate frames to choose from for validation given timespin and length"""
        for sequence in sequence_list:
            for frame_idx in range(len(sequence) - length):
                start_ts = sequence[frame_idx][TIMESTAMP_IDX]
                curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                candicate = []
                while curr_ts - start_ts <= timespin and frame_idx + 1 < len(sequence):
                    candicate.append(sequence[frame_idx])
                    frame_idx += 1
                    curr_ts = sequence[frame_idx][TIMESTAMP_IDX]
                if len(candicate) > length:
                    candidate_datalist.append(candicate)

        total_candidates = len(candidate_datalist)
        """self.logger.info(
            "total_candidates: {}, total_frames: {}, percentage: {}".format(
                total_candidates,
                total_frames,
                total_candidates / total_frames,
            )
        )"""

        """randomly sample inf_sequences from candidate_datalist"""
        sample_idx = random.sample(
            range(total_candidates), int(total_candidates * percentage)
        )
        sample_idx.sort()
        # self.logger.info(
        #     "sample size: {}, sample_idx:\n{}\n".format(len(sample_idx), sample_idx)
        # )

        for idx in sample_idx:
            val_datalist.append(candidate_datalist[idx])

        """
        update sequence_list by removing the 1st frame of each data item in val_datalist
        to avoid duplicates in train_datalist and val_datalist
        """
        for val_item in val_datalist:
            val_frame = val_item[0]
            for sequence in sequence_list:
                if val_frame in sequence:
                    sequence.remove(val_frame)

        """check datalist size"""
        total_val_datalist_frames = len(val_datalist)
        for sequence in sequence_list:
            total_sequence_frames += len(sequence)
        """self.logger.info(
            "val_datalist size: {}, current sequence size: {}, original sequence size: {}, matched: {}\n".format(
                total_val_datalist_frames,
                total_sequence_frames,
                total_frames,
                total_val_datalist_frames + total_sequence_frames == total_frames,
            )
        )"""
        assert total_val_datalist_frames + total_sequence_frames == total_frames
        return val_datalist, sequence_list

    def split_sequence(self, full_sequence_list):
        chunked_sequence_list = []
        ts_array = np.array(full_sequence_list)[:, 0].astype(np.float_)
        assert np.all(ts_array[:-1] <= ts_array[1:])

        if_consecutive = ts_array[1:] - ts_array[:-1] < self.split_timespin
        chunk_start_indices = np.where(if_consecutive == False)[0]
        # np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        chunk_start_indices += 1
        chunk_start_indices = np.insert(chunk_start_indices, 0, 0)
        chunk_start_indices = np.append(chunk_start_indices, ts_array.shape[0])

        for idx in range(chunk_start_indices.shape[0] - 1):
            chunked_sequence_list.append(
                full_sequence_list[
                    chunk_start_indices[idx] : chunk_start_indices[idx + 1]
                ]
            )

        counter = 0
        for item in chunked_sequence_list:
            counter += len(item)
        assert counter == len(full_sequence_list)
        return chunked_sequence_list

    """
    data_list items:
        [
            0  - timestamp: float

            1  - rgb cmr full path: str
            2  - map img full path: str

            3  - vehicle pose utm northing: float
            4  - vehicle pose utm easting: float
            5  - vehicle pose utm height(negative): float
            6  - vehicle pose euler angle roll: float
            7  - vehicle pose euler angle pitch: float
            8  - vehicle pose euler angle yaw: float
            9  - vehicle pose pixel coordinate x: float
            10 - vehicle pose pixel coordinate y: float
            
            11 - vehicle type code: int (0: grizzlyA, 1: grizzlyB, 2: grizzlyC)
        ]
    """

    def get_full_datalist(self):
        full_data_list = []
        gps_file = Path(self.gps_file_path)
        with open(gps_file) as file:
            lines = np.loadtxt(file, delimiter=",", dtype=np.float_)
            for curr_idx in range(len(lines)):
                full_data_list.append(self.get_data_list_from_line(lines[curr_idx]))
        return full_data_list

    def get_data_list_from_line(
        self,
        line,
    ):
        data_list = []
        img_name = str(int(line[self.TIMESTAMP_IDX])) + ".png"
        pixel_x, pixel_y = self.get_pixel_from_utm(
            line[self.UTM_NORTHING_IDX], line[self.UTM_EASTING_IDX]
        )
        data_list.extend(
            [
                line[self.TIMESTAMP_IDX],
                str(Path(self.rgb_img_dir, img_name)),
                str(Path(self.map_img_dir, img_name)),
                line[self.UTM_EASTING_IDX],
                line[3],
                line[4],
                line[5],
                line[6],
                line[7],
                pixel_x,
                pixel_y,
                int(line[self.VEHICLE_TYPE_IDX]),
            ]
        )
        return data_list

    def get_pixel_from_utm(self, utm_northing, utm_easting):
        a, d, b, e, c, f = self.jgw_info
        # a *= self.map_resize_scale
        # d *= self.map_resize_scale
        pixel_x = e * utm_easting - b * utm_northing + b * f - e * c
        pixel_x /= a * e - b * d
        pixel_y = -d * utm_easting + a * utm_northing - a * f + d * c
        pixel_y /= a * e - b * d
        assert 0 <= pixel_x < self.map_width and 0 <= pixel_y < self.map_height
        return pixel_x, pixel_y
