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


class SmallStem(torch.nn.Module):
    """Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """
    def __init__(self, patch_size:int, output_features:int):
        super(SmallStem, self).__init__()
        self.patch_size = patch_size
        self.output_features = output_features
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 96, 3, 2, 1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 192, 3, 2, 1),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(),
            torch.nn.Conv2d(192, 384, 3, 2, 1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, self.output_features, self.patch_size // 16, self.patch_size // 16, "valid")
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


class ErrorGradient(torch.nn.Module):
    def __init__(self, action_embed_dim:int, time_embed_dim:int, hidden_dim:int):
        super(ErrorGradient, self).__init__()
        self.action_project = torch.nn.Linear(action_embed_dim, hidden_dim)
        self.time_project = torch.nn.Linear(time_embed_dim, hidden_dim)
        #self.model = torch.nn.Transformer(d_model = hidden_dim)

        self.model = torch.nn.Linear(hidden_dim * 2, time_embed_dim + action_embed_dim)
        self.final_project = torch.nn.Linear(time_embed_dim + action_embed_dim, action_embed_dim)

    def forward(self, action_embed:torch.Tensor,time_embed:torch.Tensor) -> torch.Tensor:
        #print(f"action_embed {action_embed.shape}")
        #print(f"time_embed {time_embed.shape}")
        action_time_embed = torch.concat([self.action_project(action_embed), self.time_project(time_embed)], axis = -1)
        #output = self.model.forward(action_time_embed)
        #print(f"shape {action_time_embed.shape}")
        output = self.model(action_time_embed)
        return self.final_project(output)
        



class DDPM(nn.Module):
    def __init__(self, img_width, img_height, img_channel):
        super(DDPM, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.time_step = 10
        self.alpha = nn.Parameter(torch.tensor(0.97, requires_grad=False))
        self.gamma = nn.Parameter(torch.tensor(0.97), requires_grad=False)

        self.denoising = nn.Linear(self.img_channel * self.img_height * self.img_width, self.img_channel * self.img_height * self.img_width)
        #self.error_gradient = nn.Linear(self.img_channel * self.img_height * self.img_width, self.img_channel * self.img_height * self.img_width)
        self.error_gradient = ErrorGradient( self.img_channel * self.img_height * self.img_width, 1, 512)
        self.visual_encoder = SmallStem(patch_size=16, output_features=512)

    def compute_loss(self, x:torch.Tensor) -> torch.Tensor:

        visual_embedding = self.visual_encoder.forward(x)

        x = einops.rearrange(
            x,
            "batch c h w -> batch (c h w)",
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )
        batch = x.size(0)

        # assume timestamp range from [0, 1)
        timestamp = torch.rand((batch, 1), device=x.device)

        #alpha = self.alpha ** timestamp.to(x.device)
        sample_noise = torch.randn(x.shape, device=x.device)

        gradient = self.alpha * (x + self.gamma * self.error_gradient.forward(x + sample_noise, timestamp)) 

        return torch.mean((gradient - sample_noise)**2)

    def inference(self, sample_size:int) -> torch.Tensor:

        x_t = torch.randn((sample_size, self.img_width * self.img_height * self.img_channel), device = self.alpha.device)

        for t in range(self.time_step, 0, -1):

            timestamp = torch.tensor([t / self.time_step], device=self.alpha.device)
            timestamp = einops.repeat(timestamp, "one -> batch one", batch = sample_size)
            #noise = torch.randn((sample_size, self.img_width * self.img_height * self.img_channel)).to(self.alpha.device)
            x_t = self.alpha * (x_t - self.gamma * self.error_gradient.forward(x_t, timestamp))

        return einops.rearrange(
            x_t,
            "batch (c h w) -> batch c h w",
            batch=sample_size,
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )