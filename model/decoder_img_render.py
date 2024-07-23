import torch.nn as nn


class BEVImageRenderDecoder(nn.Module):
    def __init__(
        self, bev_spatial_dim, model_dim=256, hid_dim=64, logger=None, use_wandb=False
    ):
        super().__init__()
        self.logger = logger
        self.use_wandb = use_wandb

        self.decoder_block0 = nn.Sequential(
            nn.Conv2d(model_dim, hid_dim, 7, 2, 3, bias=False),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )

        self.decoder_block1 = BasicBlock(hid_dim, hid_dim, hid_dim, False)
        self.decoder_block2 = BasicBlock(hid_dim, hid_dim * 2, hid_dim * 2, True)
        self.decoder_block3 = BasicBlock(hid_dim * 2, model_dim, model_dim, True)

        self.upsample_block1 = UpSampleLayer1(
            model_dim, model_dim // 2, model_dim // 2, scale=2.0, mode="bilinear"
        )
        self.upsample_block2 = UpSampleLayer1(
            model_dim // 2, model_dim // 4, model_dim // 4, scale=2.0, mode="bilinear"
        )

        self.upsample_block4 = UpSampleLayer1(
            model_dim // 4,
            model_dim // 4,
            model_dim // 4,
            scale=2.0,
            mode="bilinear",
        )
        self.upsample_block5 = UpSampleLayer1(
            model_dim // 4,
            model_dim // 4,
            model_dim // 4,
            scale=2.0,
            mode="bilinear",
        )

        self.upsample_block3 = UpSampleLayer2(
            model_dim // 4, model_dim // 8, 3, scale=2.0, mode="bilinear"
        )

        if bev_spatial_dim == 28:
            self.decoder_layers = nn.ModuleList(
                [
                    self.decoder_block0,
                    self.decoder_block1,
                    self.decoder_block2,
                    self.decoder_block3,
                    self.upsample_block1,
                    self.upsample_block2,
                    self.upsample_block4,
                    self.upsample_block3,
                ]
            )
        elif bev_spatial_dim == 56:
            self.decoder_layers = nn.ModuleList(
                [
                    self.decoder_block0,
                    self.decoder_block1,
                    self.decoder_block2,
                    self.decoder_block3,
                    self.upsample_block1,
                    self.upsample_block2,
                    self.upsample_block3,
                ]
            )
        elif bev_spatial_dim == 14:
            self.decoder_layers = nn.ModuleList(
                [
                    self.decoder_block0,
                    self.decoder_block1,
                    self.decoder_block2,
                    self.decoder_block3,
                    self.upsample_block1,
                    self.upsample_block2,
                    self.upsample_block4,
                    self.upsample_block5,
                    self.upsample_block3,
                ]
            )

    def forward(self, x):
        for layer in self.decoder_layers:
            x = layer(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        hidden_dim,
        out_channel,
        downsample_or_not,
    ):
        super().__init__()

        if downsample_or_not:
            self.basic_block = nn.Sequential(
                nn.Conv2d(in_channel, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.Conv2d(hidden_dim, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.basic_block = nn.Sequential(
                nn.Conv2d(in_channel, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.Conv2d(hidden_dim, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(
                    hidden_dim,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),  # sigmoid
            )

    def forward(self, x):
        x = self.basic_block(x)
        return x


class UpSampleLayer1(nn.Module):
    def __init__(self, in_channel, hidden_dim, out_channel, scale, mode="bilinear"):
        super().__init__()

        self.upsample1_block = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=mode),
            nn.Conv2d(in_channel, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample1_block(x)
        return x


class UpSampleLayer2(nn.Module):
    def __init__(self, in_channel, hidden_dim, out_channel, scale, mode="bilinear"):
        super().__init__()

        self.upsample2_block = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=mode),
            nn.Conv2d(in_channel, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, out_channel, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.upsample2_block(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, model_dim=256, hid_dim=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(256, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
