import torch
from torch import nn
import einops

class VAE(nn.Module):

    def __init__(self, img_width:int, img_height:int, img_channel:int):
        super(VAE, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel

        self.latent_dim = 512

        self.encoder_mean = nn.Linear(3 * 64 * 64, self.latent_dim )
        self.encoder_std = nn.Linear(3 * 64 * 64, self.latent_dim )
        self.decoder = nn.Linear(self.latent_dim, 3 * 64 * 64)

        self.mean = nn.Parameter(torch.zeros(self.latent_dim))
        self.std = nn.Parameter(torch.ones(self.latent_dim))
        self.sample_batch = 128

    def compute_loss(self, x:torch.Tensor):
        x = einops.rearrange(
            x,
            "batch c h w -> batch (c h w)",
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )

        mean = einops.repeat(
            self.mean,
            "latent_dim ->batch sample_batch latent_dim",
            sample_batch=self.sample_batch,
            batch=x.size(0),
        )

        std = einops.repeat(
            self.std,
            "latent_dim ->batch sample_batch latent_dim",
            sample_batch=self.sample_batch,
            batch=x.size(0),
        )

        noise = torch.normal(mean = mean, std = std)

        batch_x = einops.repeat(x, "batch d -> batch sample_batch d", sample_batch=self.sample_batch)
        latent_mean = self.encoder_mean(batch_x)
        latent_std = self.encoder_std(batch_x)

        encoder_loss_1 = einops.reduce(
            einops.reduce(
                latent_std**2, "batch sample_size d ->batch sample_size", "sum"
            ),
            "batch sample_size -> batch",
            "mean",
        )
        #print(f"encoder_loss_1: {encoder_loss_1}")
        encoder_loss = 0.5 * (
            encoder_loss_1
            + einops.reduce(
                einops.reduce(
                    latent_mean**2,
                    "batch sample_size latent_dim->batch sample_size",
                    "sum",
                ),
                "batch sample_size -> batch",
                "mean",
            )
            - 2.0
            * einops.reduce(
                einops.reduce(torch.log(torch.abs(latent_std)), "b s d -> b s", "sum"),
                "b s->b",
                "mean",
            )
        )
        #print(f"encoder :{encoder_loss}")
        latent = latent_mean + latent_std * noise

        diff = batch_x - self.decoder(latent)
        decoder_loss = - 1.0 / self.sample_batch * einops.reduce(diff**2,"batch sample_size dim->batch", "sum")
        return -torch.mean(decoder_loss + encoder_loss)

    def inference(self, sample_size:int):

        mean = einops.repeat(self.mean, "latent_dim -> sample_batch latent_dim", sample_batch = sample_size)
        std = einops.repeat(self.std, "latent_dim -> sample_batch latent_dim", sample_batch = sample_size)
        noise = torch.normal(mean = mean, std = std)
        generate = self.decoder(noise)

        return einops.rearrange(
            generate,
            "batch (c h w) -> batch c h w",
            batch=sample_size,
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )

class DDPM(nn.Module):
    def __init__(self, img_width, img_height, img_channel):
        super(DDPM, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.time_step = 10
        self.alpha = nn.Parameter(torch.tensor(0.97, requires_grad=False))

        self.denoising = nn.Linear(self.img_channel * self.img_height * self.img_width, self.img_channel * self.img_height * self.img_width)
        self.error_gradient = nn.Linear(self.img_channel * self.img_height * self.img_width, self.img_channel * self.img_height * self.img_width)

    def compute_loss(self, x:torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(
            x,
            "batch c h w -> batch (c h w)",
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )
        batch = x.size(0)
        timestamp = torch.linspace(0, self.time_step - 1, 1)
        timestamp = einops.repeat(timestamp, "scalar -> batch scalar", batch = batch, scalar = 1)
        alpha = self.alpha ** timestamp.to(x.device)
        sample_noise = torch.normal(mean = torch.zeros(x.shape), std = torch.ones(x.shape)).to(x.device)

        sample = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * sample_noise

        return torch.mean(self.error_gradient(sample) - sample_noise)**2

    def inference(self, sample_size:int) -> torch.Tensor:

        x_t = torch.normal(mean = torch.zeros((sample_size, self.img_width * self.img_height * self.img_channel))).to(self.alpha.device)

        for t in range(self.time_step, 0, -1):
            noise = torch.normal(mean = torch.zeros((sample_size, self.img_width * self.img_height * self.img_channel))).to(self.alpha.device)
            x_t = (
                self.alpha**-0.5 * (x_t - (1 - self.alpha) / (1 - self.alpha**t) * self.error_gradient(x_t))
                + (1 - self.alpha)
                * torch.sqrt(self.alpha ** (t - 1))
                / ((1 - self.alpha**t))
                * noise
            )

        return einops.rearrange(
            x_t,
            "batch (c h w) -> batch c h w",
            batch=sample_size,
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )