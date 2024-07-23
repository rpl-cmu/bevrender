import os
import time
import json
import wandb
import torch
import logging
import calendar
from pathlib import Path
from einops import rearrange


def wandb_log_data(
    type,
    epoch,
    index,
    jump_frame,
    sample_loss,
    model_output,
    map_tensor,
    camera_tensor,
    norm=None,
):
    if index % jump_frame == 0:
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
        log_image = torch.cat((log_img_cmr, log_image), axis=1)
        if type == "train":
            wandb.log(
                {
                    "sample_loss": sample_loss,
                    "gradient_norm": norm,
                    "image": wandb.Image(log_image, caption=f"log image {index}"),
                    "epoch": epoch,
                }
            )
        else:
            wandb.log(
                {
                    "val_sample_loss": sample_loss,
                    "val_image": wandb.Image(log_image, caption=f"log image {index}"),
                    "epoch": epoch,
                },
            )
    else:
        if type == "train":
            wandb.log(
                {
                    "sample_loss": sample_loss,
                    "gradient_norm": norm,
                    "epoch": epoch,
                }
            )
        else:
            wandb.log(
                {
                    "val_sample_loss": sample_loss,
                    "epoch": epoch,
                }
            )


def count_parameters(model, logger):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = params / 1000000
    logger.info(f"model parameters : {num_params}M\n")


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)-22s:%(lineno)3d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger


def get_save_name(config, save_params):
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    save_name = str(Path(config["CKPT_DIR"], str(ts)))
    os.makedirs(save_name, exist_ok=True)
    if save_params:
        with open(os.path.join(save_name, f"parameters.json"), "w") as outfile:
            json.dump(vars(config), outfile, indent=4)
    return save_name


def save_model(
    savePath,
    camera_encoder,
    map_encoder,
    optimizer,
    scheduler,
    epoch,
    best=False,
):
    if best:
        model_name = f"best_epoch_{epoch}.pth"
    else:
        model_name = "last_epoch.pth"

    if map_encoder:
        torch.save(
            {
                "epoch": epoch,
                "camera_encoder_state_dict": camera_encoder.state_dict(),
                "map_encoder_state_dict": map_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            Path(savePath, model_name),
        )
    else:
        torch.save(
            {
                "epoch": epoch,
                "camera_encoder_state_dict": camera_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            Path(savePath, model_name),
        )
