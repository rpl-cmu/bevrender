import os
import torch
import wandb
import torchvision
import numpy as np
from pathlib import Path
from einops import rearrange
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.model_selection import KFold, train_test_split
import torchvision.transforms.functional as F
from utils.scheduler import WarmupCosineSchedule
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from loss.mse_loss import MSELoss
from loss.l1_loss import L1Loss
from loss.cross_entropy_loss import CrossEntropyLoss
from loss.lift_loss import LiftedStructureLoss
from loss.triplet_loss_metric import TripletLossMetricLearning
from loss.contrastive_loss import ContrastiveLoss
from model.bevrender import BEVRender
from configuration.config import get_config, save_config_given_dir
from utils.utils import get_logger, get_save_name, save_model, count_parameters
from dataloader.dataprocessor import DatasetProcessor


def ddp_setup(rank, world_size):
    init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )


class Trainer:
    # model_output_dim = 64 * 56 * 56
    model_output_dim = 64 * 28 * 28

    def __init__(
        self,
        camera_encoder: torch.nn.Module,
        map_encoder: torch.nn.Module,
        train_val_dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        k_fold: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        total_epochs: int,
        gpu_id: int,
        device,
        log_train_img_batch_frequency: int,
        log_val_img_batch_frequency: int,
        val_frequency: int,
        val_metric: str,
        save_val_results: bool,
        work_dir: str,
        distributed: bool,
        save_ckpt: bool,
        loss_type: str,
        seed,
        logger=None,
        wandb_run=None,
    ) -> None:
        self.gpu_id = gpu_id
        self.device = device
        self.train_val_dataset = train_val_dataset
        self.k_fold = k_fold
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.work_dir = work_dir
        self.logger = logger
        self.wandb_run = wandb_run
        self.loss_type = loss_type
        self.val_metric = val_metric
        self.distributed = distributed
        self.save_ckpt = save_ckpt
        self.save_val_results = save_val_results

        self.log_train_img_batch_frequency = log_train_img_batch_frequency
        self.log_val_img_batch_frequency = log_val_img_batch_frequency
        self.val_frequency = val_frequency

        """record best epoch and loss"""
        self.best_epoch = 0
        self.best_epoch_loss = 1e8
        self.best_epoch_recall = 0.0
        self.total_epochs = total_epochs

        """set up training mode and loss type"""
        self.image_rendering = False
        self.image_retrieval = False
        if (
            "MSE" in loss_type
            or "L1" in loss_type
            or "CROSS_ENTROPY_RENDER" in loss_type
        ):
            self.image_rendering = True
        if (
            "LIFT" in loss_type
            or "TRIPLET" in loss_type
            or "CONTRASTIVE" in loss_type
            or "CROSS_ENTROPY_RTRVL" in loss_type
        ):
            self.image_retrieval = True

        if "MSE" in loss_type:
            self.image_rendering_loss = MSELoss()
        elif "L1" in loss_type:
            self.image_rendering_loss = L1Loss()
        elif "CROSS_ENTROPY_RENDER" in loss_type:
            self.image_rendering_loss = CrossEntropyLoss()
        if "LIFT" in loss_type:
            self.image_retrieval_loss = LiftedStructureLoss()
        elif "TRIPLET" in loss_type:
            self.image_retrieval_loss = TripletLossMetricLearning()
        elif "CONTRASTIVE" in loss_type:
            self.image_retrieval_loss = ContrastiveLoss()
        elif "CROSS_ENTROPY_RTRVL" in loss_type:
            self.image_retrieval_loss = CrossEntropyLoss()

        """set up camera encoder and map encoder"""
        if distributed:
            camera_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                camera_encoder
            )
            self.camera_encoder = camera_encoder.to(device)
            self.camera_encoder = DDP(
                camera_encoder, device_ids=[gpu_id], find_unused_parameters=True
            )
            if map_encoder:
                map_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(map_encoder)
                self.map_encoder = map_encoder.to(device)
                self.map_encoder = DDP(
                    map_encoder, device_ids=[gpu_id], find_unused_parameters=True
                )
        else:
            self.camera_encoder = camera_encoder.to(device)
            self.map_encoder = map_encoder.to(device) if map_encoder else None

        """set up batch size"""
        self.train_num_batches = 0
        self.val_num_batches = 0

        """set up total loss"""
        self.tr_epoch_render_loss = 0.0
        self.tr_epoch_retrieval_loss = 0.0
        self.tr_epoch_loss = 0.0
        self.val_epoch_render_loss = 0.0
        self.val_epoch_retrieval_loss = 0.0
        self.val_epoch_loss = 0.0
        self.val_epoch_recall_1 = 0.0
        self.val_epoch_recall_5 = 0.0
        self.val_epoch_recall_10 = 0.0

    def _run_epoch(
        self, epoch: int, fold: int, train_loader, val_loader, apply_validation: bool
    ):
        self.tr_epoch_render_loss = 0.0
        self.tr_epoch_retrieval_loss = 0.0
        self.tr_epoch_loss = 0.0
        recall_1, recall_5, recall_10 = 0.0, 0.0, 0.0
        (
            self.logger.info(
                "Training epoch {}, fold {}, training dataset: {}, validation dataset: {}".format(
                    epoch,
                    fold,
                    len(train_loader) * self.batch_size,
                    len(val_loader) * self.batch_size,
                )
            )
            if self.gpu_id == 0
            else None
        )
        if self.distributed:
            train_loader.sampler.set_epoch(epoch)

        """training loop"""
        for tr_idx, batch in enumerate(train_loader):
            wandb_tr_dict = {}
            tr_batch_loss = 0.0
            """
            cmr_tensor: (bs, 6, 3, 512, 640)
            map_tensor: (bs, 3, 224, 224)
            veh_pose:   (bs, 6, 3)
            veh_type:   (bs, 1)
            timestamp:  (bs)
            """
            (ori_cmr_tensor, ori_map_tensor, veh_pose, veh_type, timestamp) = (
                batch["camera"].to(self.gpu_id),
                batch["map"].to(self.gpu_id),
                batch["vehicle_pose"].to(self.gpu_id),
                batch["vehicle_type"].to(self.gpu_id),
                batch["timestamp"],
            )
            self.optimizer.zero_grad()

            """get model outputs"""
            camera_tensor, wandb_tr_dict = self.camera_encoder(
                ori_cmr_tensor, veh_pose, veh_type, wandb_tr_dict, return_wandb_log=True
            )
            map_tensor, wandb_tr_dict = (
                self.map_encoder(ori_map_tensor, wandb_tr_dict, return_wandb_log=False)
                if self.map_encoder
                else (ori_map_tensor, wandb_tr_dict)
            )
            assert camera_tensor.shape == map_tensor.shape

            """add up losses for image rendering and retrieval"""
            if self.image_rendering:
                tr_batch_rendering_loss = self.image_rendering_loss.get_loss(
                    camera_tensor, map_tensor
                )
                tr_batch_loss += tr_batch_rendering_loss
                avg_batch_render_loss = tr_batch_rendering_loss / self.train_num_batches
                self.tr_epoch_render_loss += avg_batch_render_loss
                self.tr_epoch_loss += avg_batch_render_loss
            if self.image_retrieval:
                tr_batch_retrieval_loss = self.image_retrieval_loss.get_loss(
                    camera_tensor, map_tensor
                )
                tr_batch_loss += tr_batch_retrieval_loss
                avg_batch_retrieval_loss = (
                    tr_batch_retrieval_loss / self.train_num_batches
                )
                self.tr_epoch_retrieval_loss += avg_batch_retrieval_loss
                self.tr_epoch_loss += avg_batch_retrieval_loss

            """backpropagation"""
            tr_batch_loss.backward()
            camera_grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.camera_encoder.parameters(), 1.0)
                if self.camera_encoder
                else None
            )
            map_grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.map_encoder.parameters(), 1.0)
                if self.map_encoder
                else None
            )

            self.optimizer.step()

            """log training for gpu 0 if distributed or for single gpu training"""
            if (self.distributed and self.gpu_id == 0) or (not self.distributed):
                self.log_batch(
                    idx=tr_idx,
                    render_loss=(
                        tr_batch_rendering_loss if self.image_rendering else None
                    ),
                    retrieval_loss=(
                        tr_batch_retrieval_loss if self.image_retrieval else None
                    ),
                    total_loss=tr_batch_loss,
                    camera_grad_norm=camera_grad_norm,
                    map_grad_norm=map_grad_norm,
                    num_batches=self.train_num_batches,
                )

                """set up wandb log dictionary"""
                if self.wandb_run:
                    wandb_tr_dict["train_batch_loss"] = tr_batch_loss
                    wandb_tr_dict["learning_rate"] = self.scheduler.get_last_lr()[0]
                    wandb_tr_dict["epoch"] = epoch

                    if self.image_rendering:
                        wandb_tr_dict["train_batch_render_loss"] = (
                            tr_batch_rendering_loss
                        )
                    if self.image_retrieval:
                        wandb_tr_dict["train_batch_retrieval_loss"] = (
                            tr_batch_retrieval_loss
                        )
                    wandb_tr_dict["camera_encoder_grad_norm"] = camera_grad_norm
                    if self.map_encoder:
                        wandb_tr_dict["map_encoder_grad_norm"] = map_grad_norm

                    if (
                        self.image_rendering
                        and tr_idx % self.log_train_img_batch_frequency == 0
                    ):
                        wandb_tr_dict["train_image"] = wandb.Image(
                            self.get_log_image(
                                camera_tensor[0],
                                ori_map_tensor[0],
                                ori_cmr_tensor[0, -1, ...],
                            ),
                            caption=f"train epoch {epoch} - {timestamp[0]}",
                        )

                    if tr_idx == self.train_num_batches - 1:
                        if self.image_rendering:
                            wandb_tr_dict["train_epoch_render_loss"] = (
                                self.tr_epoch_render_loss
                            )
                        if self.image_retrieval:
                            wandb_tr_dict["train_epoch_retrieval_loss"] = (
                                self.tr_epoch_retrieval_loss
                            )
                        wandb_tr_dict["train_epoch_loss"] = self.tr_epoch_loss

                    self.wandb_run.log(
                        wandb_tr_dict,
                    )

        """validation loop"""
        if apply_validation and (epoch + 1) % self.val_frequency == 0:
            self.logger.info(f"Validation epoch {epoch}") if self.gpu_id == 0 else None

            self.val_epoch_render_loss = 0.0
            self.val_epoch_retrieval_loss = 0.0
            self.val_epoch_loss = 0.0
            self.val_epoch_recall_1 = 0.0
            self.val_epoch_recall_5 = 0.0
            self.val_epoch_recall_10 = 0.0

            """if self.distributed:
                val_loader.sampler.set_epoch(epoch)"""

            if self.image_retrieval:
                global_camera_tensor, global_map_tensor = np.zeros(
                    (self.batch_size * self.val_num_batches, self.model_output_dim)
                ), np.zeros(
                    (self.batch_size * self.val_num_batches, self.model_output_dim)
                )

            self.camera_encoder.eval()
            if self.map_encoder:
                self.map_encoder.eval()

            with torch.no_grad():
                for val_idx, batch in enumerate(val_loader):
                    wandb_val_dict = {}
                    val_batch_loss = 0.0
                    (
                        ori_cmr_tensor,
                        ori_map_tensor,
                        veh_pose,
                        veh_type,
                        timestamp,
                    ) = (
                        batch["camera"].to(self.gpu_id),
                        batch["map"].to(self.gpu_id),
                        batch["vehicle_pose"].to(self.gpu_id),
                        batch["vehicle_type"].to(self.gpu_id),
                        batch["timestamp"],
                    )

                    """get model outputs"""
                    camera_tensor, wandb_val_dict = self.camera_encoder(
                        ori_cmr_tensor,
                        veh_pose,
                        veh_type,
                        wandb_val_dict,
                        return_wandb_log=True,
                    )
                    map_tensor, wandb_val_dict = (
                        self.map_encoder(
                            ori_map_tensor, wandb_val_dict, return_wandb_log=False
                        )
                        if self.map_encoder
                        else (ori_map_tensor, wandb_val_dict)
                    )

                    """add up losses for image rendering"""
                    if self.image_rendering:
                        val_batch_rendering_loss = self.image_rendering_loss.get_loss(
                            camera_tensor, map_tensor
                        )
                        val_batch_loss += val_batch_rendering_loss
                        avg_batch_render_loss = (
                            val_batch_rendering_loss / self.val_num_batches
                        )
                        self.val_epoch_render_loss += avg_batch_render_loss
                        self.val_epoch_loss += avg_batch_render_loss

                    """add up losses for image retrieval"""
                    if self.image_retrieval:
                        global_camera_tensor[
                            val_idx * self.batch_size : (val_idx + 1) * self.batch_size,
                            :,
                        ] = (
                            camera_tensor.detach().cpu().numpy()
                        )
                        global_map_tensor[
                            val_idx * self.batch_size : (val_idx + 1) * self.batch_size,
                            :,
                        ] = (
                            map_tensor.detach().cpu().numpy()
                        )
                        val_batch_retrieval_loss = self.image_retrieval_loss.get_loss(
                            camera_tensor, map_tensor
                        )
                        val_batch_loss += val_batch_retrieval_loss
                        avg_batch_retrieval_loss = (
                            val_batch_retrieval_loss / self.val_num_batches
                        )
                        self.val_epoch_retrieval_loss += avg_batch_retrieval_loss
                        self.val_epoch_loss += avg_batch_retrieval_loss

                    """log validation"""
                    if self.gpu_id == 0:
                        self.log_batch(
                            idx=val_idx,
                            render_loss=(
                                val_batch_rendering_loss
                                if self.image_rendering
                                else None
                            ),
                            retrieval_loss=(
                                val_batch_retrieval_loss
                                if self.image_retrieval
                                else None
                            ),
                            total_loss=val_batch_loss,
                            num_batches=self.val_num_batches,
                        )

                        """set up wandb log dictionary"""
                        if self.wandb_run:
                            wandb_val_dict["val_batch_loss"] = val_batch_loss
                            wandb_val_dict["epoch"] = epoch

                            if self.image_rendering:
                                wandb_val_dict["val_batch_render_loss"] = (
                                    val_batch_rendering_loss
                                )
                            if self.image_retrieval:
                                wandb_val_dict["val_batch_retrieval_loss"] = (
                                    val_batch_retrieval_loss
                                )

                            if (
                                self.image_rendering
                                and val_idx % self.log_val_img_batch_frequency == 0
                            ):
                                wandb_val_dict["val_image"] = wandb.Image(
                                    self.get_log_image(
                                        camera_tensor[0],
                                        ori_map_tensor[0],
                                        ori_cmr_tensor[0, -1, ...],
                                    ),
                                    caption=f"validation epoch {epoch} - {timestamp[0]}",
                                )

                            if val_idx == self.val_num_batches - 1:
                                wandb_val_dict["val_epoch_loss"] = self.val_epoch_loss
                                if self.image_retrieval:
                                    """calculating & logging retrieval recall"""
                                    recall_1, recall_5, recall_10 = self.get_recall(
                                        global_camera_tensor, global_map_tensor
                                    )
                                    self.val_epoch_recall_1 = recall_1
                                    self.val_epoch_recall_5 = recall_5
                                    self.val_epoch_recall_10 = recall_10

                                    wandb_val_dict["val_R@1"] = recall_1
                                    wandb_val_dict["val_R@5"] = recall_5
                                    wandb_val_dict["val_R@10"] = recall_10

                            self.wandb_run.log(
                                wandb_val_dict,
                            )

            if self.val_metric == "LOSS":
                if self.val_epoch_loss < self.best_epoch_loss:
                    self.best_epoch_loss = self.val_epoch_loss
                    self.best_epoch = epoch
                    (
                        self.save_checkpoint(epoch, best=True)
                        if self.save_ckpt and self.gpu_id == 0
                        else None
                    )
                    (
                        self.save_val_images(epoch, val_loader)
                        if self.save_val_results
                        else None
                    )
                else:
                    (
                        self.save_checkpoint(epoch, best=False)
                        if self.save_ckpt and self.gpu_id == 0
                        else None
                    )

            elif self.val_metric == "RECALL":
                if self.val_epoch_recall_5 > self.best_epoch_recall:
                    self.best_epoch_recall = self.val_epoch_recall_5
                    self.best_epoch = epoch
                    (
                        self.save_checkpoint(epoch, best=True)
                        if self.save_ckpt and self.gpu_id == 0
                        else None
                    )
                else:
                    (
                        self.save_checkpoint(epoch, best=False)
                        if self.save_ckpt and self.gpu_id == 0
                        else None
                    )

            self.camera_encoder.train()
            self.map_encoder.train() if self.map_encoder else None

        if self.distributed:
            dist.barrier()
        self.scheduler.step()

        if (
            apply_validation
            and (epoch + 1) % self.val_frequency == 0
            and self.gpu_id == 0
        ):
            self.logger.info(
                "Summary of epoch {}/{} at GPU {} - training loss: {:4.8f},  validation loss: {:4.8f}".format(
                    epoch,
                    self.total_epochs,
                    self.gpu_id,
                    self.tr_epoch_loss,
                    self.val_epoch_loss,
                )
            )
            if self.image_retrieval:
                self.logger.info(
                    "Summary of epoch {}/{} at GPU {} - R@1-{:2.2f}%,  R@5-{:2.2f}%,  R@10-{:2.2f}%".format(
                        epoch,
                        self.total_epochs,
                        self.gpu_id,
                        recall_1,
                        recall_5,
                        recall_10,
                    )
                )
        else:
            (
                self.logger.info(
                    "Summary of epoch {}/{} at GPU {} - training loss: {:.8f}".format(
                        epoch, self.total_epochs, self.gpu_id, self.tr_epoch_loss
                    )
                )
                if self.gpu_id == 0
                else None
            )
        self.logger.info("") if self.gpu_id == 0 else None

    def get_recall(self, global_camera_tensor, global_map_tensor):
        recall_1, recall_5, recall_10 = 0.0, 0.0, 0.0

        dist_array = 2.0 - 2.0 * np.matmul(global_camera_tensor, global_map_tensor.T)
        length_recall_array = 11
        val_accuracy = np.zeros(length_recall_array)
        for i in range(length_recall_array):
            accuracy = 0.0
            data_amount = 0.0
            for k in range(dist_array.shape[0]):
                gt_dist = dist_array[k, k]
                prediction = np.sum(dist_array[:, k] < gt_dist)
                if prediction < i:
                    accuracy += 1.0
                data_amount += 1.0
            accuracy /= data_amount
            val_accuracy[i] = accuracy

        recall_1 = val_accuracy[1] * 100
        recall_5 = val_accuracy[5] * 100
        recall_10 = val_accuracy[10] * 100
        return recall_1, recall_5, recall_10

    def log_batch(
        self,
        idx,
        total_loss,
        num_batches,
        render_loss=None,
        retrieval_loss=None,
        camera_grad_norm=None,
        map_grad_norm=None,
    ):
        log_string = "step: {i:3d}/{len:3d},".format(i=idx, len=num_batches)
        if self.image_rendering:
            log_string += f" render_ls {render_loss:4.6f},"
        if self.image_retrieval:
            log_string += f" retrvl_ls {retrieval_loss:4.6f},"
        log_string += f" total_ls {total_loss:4.6f},"
        # log_string += f" cuda {cuda_memory:2.4f}GB,"
        if camera_grad_norm:
            log_string += f" cmr_grad {camera_grad_norm:6.4f},"
        if map_grad_norm:
            log_string += f" map_grad {map_grad_norm:6.4f}"
        self.logger.info(log_string)

    def save_checkpoint(self, epoch, best=False):
        save_model(
            savePath=self.work_dir,
            camera_encoder=self.camera_encoder,
            map_encoder=self.map_encoder if self.map_encoder else None,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best=best,
        )
        self.logger.info(f"model saved at epoch {epoch} and GPU {self.gpu_id}")

    def save_val_images(self, epoch, val_loader):
        val_image_dir = Path(self.work_dir, "best_epoch_val".format(self.gpu_id))
        os.makedirs(val_image_dir, exist_ok=True)

        for batch in val_loader:
            ori_cmr_tensor, veh_pose, veh_type, timestamp = (
                batch["camera"].to(self.gpu_id),
                batch["vehicle_pose"].to(self.gpu_id),
                batch["vehicle_type"].to(self.gpu_id),
                batch["timestamp"],
            )
            camera_tensor, _ = self.camera_encoder(
                ori_cmr_tensor, veh_pose, veh_type, None, return_wandb_log=False
            )
            for output, ts in zip(camera_tensor, timestamp):
                torchvision.utils.save_image(
                    output,
                    Path(
                        val_image_dir,
                        f"{ts}.png",
                    ),
                )
        (
            self.logger.info(
                "image saved at epoch {} and GPU {}".format(epoch, self.gpu_id)
            )
            if self.gpu_id == 0
            else None
        )

    def get_log_image(self, model_output, map_tensor, camera_tensor):
        log_img_map = (map_tensor - map_tensor.min()) / (
            map_tensor.max() - map_tensor.min()
        )
        log_image = torch.cat(
            (log_img_map, torch.zeros_like(log_img_map), model_output),
            axis=2,
        )
        log_img_cmr = rearrange(
            (camera_tensor - camera_tensor.min())
            / (camera_tensor.max() - camera_tensor.min()),
            "b c h w -> c h (b w)",
        )
        log_img_cmr = F.resize(log_img_cmr, (224, 672))
        log_image = torch.cat((log_img_cmr, log_image), axis=1)
        return log_image

    def train(self, apply_validation: bool):
        num_epoch = 0
        epoch_per_fold = 10
        while num_epoch + 1 < self.total_epochs:
            kfold = KFold(n_splits=self.k_fold, shuffle=True)
            for fold, (train_index, val_index) in enumerate(
                kfold.split(self.train_val_dataset)
            ):
                train_dataset = Subset(self.train_val_dataset, train_index)
                val_dataset = Subset(self.train_val_dataset, val_index)

                if self.distributed:
                    train_sampler = DistributedSampler(train_dataset, shuffle=True)
                    val_sampler = DistributedSampler(val_dataset, shuffle=False)
                else:
                    train_sampler = None
                    val_sampler = None

                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=self.batch_size,
                    sampler=train_sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=self.batch_size,
                    sampler=val_sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

                for _ in range(epoch_per_fold):
                    self.train_num_batches = len(train_loader)
                    self.val_num_batches = len(val_loader)
                    self._run_epoch(
                        num_epoch, fold, train_loader, val_loader, apply_validation
                    )
                    num_epoch += 1


def load_train_objs(config, logger):
    model = BEVRender(config, logger, mode="train")
    model_parameters = list(model.parameters())

    map_encoder = None

    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
        eps=config["EPS"],
    )
    """
    scheduler - WarmupCosineSchedule
        warpmup_steps:  5
        t_total:        config["TOTAL_EPOCHS"]
        cycles:         0.5
        last_epoch:     -1
        
    meaning - Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    
    [
        increase from 0 to 1 within 5 steps, 
        then decrease from 1 to 0 within (total_epoch - 5) steps
    ]
    """
    scheduler = WarmupCosineSchedule(optimizer, 5, config["TOTAL_EPOCHS"])
    return model, map_encoder, optimizer, scheduler


def process_train(
    rank: int,
    world_size: int,
    distributed: bool,
    ckpt_dir: str,
    config,
):
    if distributed:
        ddp_setup(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda", rank if torch.cuda.is_available() else "cpu")

    logger = get_logger()
    wandb_run = wandb.init(project="bev") if config["USE_WANDB"] and rank == 0 else None

    camera_encoder, map_encoder, optimizer, scheduler = load_train_objs(config, logger)
    count_parameters(camera_encoder, logger) if rank == 0 else None
    jgw_info = config["MAP_JGW_INFO"]

    data_processor = DatasetProcessor(
        dataset_dir=config["DATASET_DIR"],
        overlap=config["OVERLAP"],
        distributed=distributed,
        k_fold=config["K_FOLD"],
        window_timespin=config["WINDOW_TIMESPIN"] * 1e6,
        window_num_imgs=config["WINDOW_NUM_IMGS"],
        batch_size=config["BATCH_SIZE"],
        num_views=config["NUM_VIEWS"],
        num_workers=config["NUM_WORKERS"],
        pin_memory=config["PIN_MEMORY"],
        resize_cmr_img=config["RESIZE_IMG"],
        resize_img_height=config["RESIZE_IMG_HEIGHT"],
        resize_img_width=config["RESIZE_IMG_WIDTH"],
        img_norm_mean=config["CAMERA_NORM_MEAN"],
        img_norm_std=config["CAMERA_NORM_STD"],
        map_norm_mean=config["MAP_NORM_MEAN"],
        map_norm_std=config["MAP_NORM_STD"],
        gps_file_path=config["GPS_FILE_PATH"],
        rgb_img_dir=config["RGB_IMG_DIR"],
        map_img_dir=config["MAP_IMG_DIR"],
        map_width=config["MAP_WIDTH"],
        map_height=config["MAP_HEIGHT"],
        map_resize_scale=config["MAP_RESIZE_SCALE"],
        jgw_info=jgw_info,
        logger=logger,
    )
    full_dataset = data_processor.process_dataset()
    dataset_length = len(full_dataset)

    if config["SPLIT_INF_SET"]:
        dist.barrier() if distributed else None

        indices = np.arange(dataset_length)
        train_indices, inf_indices = train_test_split(
            indices, test_size=config["INF_SET_RATIO"], random_state=config["SEED"]
        )
        train_val_dataset = Subset(full_dataset, train_indices)
        inf_dataset = Subset(full_dataset, inf_indices)

        if rank == 0:
            logger.info("backbone architecture: {}".format(config["DAT_BACKBONE_TYPE"]))
            logger.info(
                "training set {}, inference set {}".format(
                    len(train_val_dataset), len(inf_dataset)
                )
            )
            inf_set_save = {
                "datalist": [inf_dataset.dataset.datalist[i] for i in inf_indices]
            }
            torch.save(inf_set_save, Path(ckpt_dir, "inference_dataset.pth"))

        dist.barrier() if distributed else None
    else:
        train_val_dataset, inf_dataset = full_dataset, None

    trainer = Trainer(
        camera_encoder=camera_encoder,
        map_encoder=map_encoder,
        train_val_dataset=train_val_dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
        pin_memory=config["PIN_MEMORY"],
        k_fold=config["K_FOLD"],
        optimizer=optimizer,
        scheduler=scheduler,
        total_epochs=config["TOTAL_EPOCHS"],
        gpu_id=rank,
        device=device,
        log_train_img_batch_frequency=config["WANDB_LOG_IMG_FERQ_TRAIN"],
        log_val_img_batch_frequency=config["WANDB_LOG_IMG_FERQ_VAL"],
        val_frequency=config["VALIDATION_FREQUENCY"],
        val_metric=config["VALIDATION_METRIC"],
        save_val_results=config["SAVE_VAL_RESULTS"],
        work_dir=ckpt_dir,
        distributed=distributed,
        save_ckpt=config["SAVE_CKPT"],
        loss_type=config["LOSS_TYPE"],
        seed=config["SEED"],
        logger=logger,
        wandb_run=wandb_run,
    )
    trainer.train(
        apply_validation=config["APPLY_VALIDATION"],
    )
    if distributed:
        destroy_process_group()


def main(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    logger = get_logger()
    config = get_config(print_or_not=True, save_or_not=False)
    ckpt_dir = get_save_name(config=config, save_params=False)
    save_config_given_dir(config, ckpt_dir)

    """set up seeds"""
    torch.manual_seed(config["SEED"])
    np.random.seed(config["SEED"])

    logger.info("Working directory: {}".format(ckpt_dir))
    logger.info("Loss type: {}".format(config["LOSS_TYPE"]))

    if config["DISTRIBUTED_TRAINING"]:
        logger.info(
            "Distributed training starts, number of GPUs used: {}".format(world_size)
        )
        mp.spawn(
            process_train,
            args=(
                world_size,
                True,
                ckpt_dir,
                config,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        logger.info("Single GPU training starts....")
        process_train(
            rank=0,
            world_size=1,
            distributed=False,
            ckpt_dir=ckpt_dir,
            config=config,
        )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    # world_size = 2
    main(world_size)
