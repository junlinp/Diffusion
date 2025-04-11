import torch
from torch import nn
import einops
import math
from accelerate import Accelerator

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

class FeedForward(torch.nn.Module):
    def __init__(self, feature_num:int, dropout:int = 0.1):
        super(FeedForward, self).__init__()

        self.first_project = nn.Linear(feature_num, 2 * feature_num)
        self.activation = nn.SiLU()
        self.dropoutp = nn.Dropout(dropout)
        self.final_project = nn.Linear(2 * feature_num, feature_num)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        input = x
        x = self.first_project(x)
        x = self.activation(x)
        x = self.dropoutp(x)
        x = self.final_project(x)
        return x + input


class MultipleHeadAttention(torch.nn.Module):
    def __init__(self, d_model:int, n_head:int):
        super(MultipleHeadAttention, self).__init__()
        self.total_hidden_dim = 512
        assert self.total_hidden_dim % n_head == 0
        self.n_head = n_head
        self.d_head = self.total_hidden_dim // self.n_head
        self.q_w = torch.nn.Linear(d_model, self.total_hidden_dim)
        self.k_w = torch.nn.Linear(d_model, self.total_hidden_dim) 
        self.v_w = torch.nn.Linear(d_model, self.total_hidden_dim) 
        self.layer_norm = torch.nn.LayerNorm(self.total_hidden_dim)

        self.feedforward = FeedForward(self.total_hidden_dim)
        self.final_project = nn.Linear(self.total_hidden_dim, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)

        q = einops.rearrange(q, "batch seq (head_num head_dim) -> batch seq head_num head_dim", head_num = self.n_head, head_dim = self.d_head)
        k = einops.rearrange(k, "batch seq (head_num head_dim) -> batch seq head_num head_dim", head_num = self.n_head, head_dim = self.d_head)
        v = einops.rearrange(v, "batch seq (head_num head_dim) -> batch seq head_num head_dim", head_num = self.n_head, head_dim = self.d_head)

        attention = einops.einsum(q, k, "batch seq1 head_num head_dim, batch seq2 head_num head_dim -> batch head_num seq1 seq2") / math.sqrt(self.d_head)
        attention = torch.softmax(attention, -1)

        o = einops.einsum(attention, v, "batch head_num seq1 seq2, batch seq2 head_num head_dim->batch seq1 head_num head_dim")
        o = einops.rearrange(o, "batch seq head_num head_dim -> batch seq (head_num head_dim)")
        o = self.layer_norm(o)

        return x + self.final_project(self.feedforward.forward(o))



class Transformer(torch.nn.Module):
    def __init__(self, input_feature_dim:int, d_model:int, block_num:int):
        super(Transformer, self).__init__()

        self.model = torch.nn.ModuleList([MultipleHeadAttention(d_model, 8) for i in range(block_num)])
        self.layer_norm = torch.nn.LayerNorm(input_feature_dim) 

    def forward(self, x:torch.Tensor)->torch.Tensor:
        for lyr in self.model:
            x = lyr.forward(x)
        return x

class ErrorGradient(torch.nn.Module):
    def __init__(self, action_embed_dim:int, time_embed_dim:int, hidden_dim:int):
        super(ErrorGradient, self).__init__()
        self.time_project = torch.nn.Linear(time_embed_dim, action_embed_dim)
        self.model = Transformer(action_embed_dim, action_embed_dim, 4)
        self.layer_norm = torch.nn.LayerNorm(action_embed_dim)

    def forward(self, action_embed:torch.Tensor,time_embed:torch.Tensor) -> torch.Tensor:
        action_time_embed = action_embed + self.time_project(time_embed)
        output = self.model(action_time_embed)
        return self.layer_norm(output + action_time_embed)
        



class DDPM(nn.Module):
    def __init__(self, img_width, img_height, img_channel):
        super(DDPM, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.time_step = 1024
        self.sample_size = 64

        #self.denoising = nn.Linear(self.img_channel * self.img_height * self.img_width, self.img_channel * self.img_height * self.img_width)
        #self.error_gradient = nn.Linear(self.img_channel * self.img_height * self.img_width, self.img_channel * self.img_height * self.img_width)
        self.error_gradient = ErrorGradient( self.img_channel * self.img_height * self.img_width, 1, 512)
        #self.visual_encoder = SmallStem(patch_size=16, output_features=512)

        self.beta_0 = 1e-4
        self.beta_t = 0.02
        self.alpha_buffer = [1.0 - (i / self.time_step * (self.beta_t - self.beta_0) + self.beta_0) for i in range(self.time_step)]
        base = 1.0
        self.alpha_cumsum_buffer = []
        for alpha in self.alpha_buffer:
            self.alpha_cumsum_buffer.append(base * alpha)
            base = base * alpha

    def compute_loss(self, x:torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(
            x,
            "batch c h w -> batch (c h w)",
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )
        x = einops.repeat(x, "batch features -> batch sample_size features", sample_size = self.sample_size)
        batch = x.size(0)

        timestamp = (torch.rand((batch, self.sample_size, 1), device=x.device) * self.sample_size).to(torch.int32)
        alpha_cumsum = torch.tensor(self.alpha_cumsum_buffer,device=x.device)[timestamp]
        sample_noise = torch.randn(x.shape, device=x.device)
        gradient = self.error_gradient.forward(torch.sqrt(alpha_cumsum) * x + torch.sqrt(1 - alpha_cumsum) * sample_noise, timestamp.to(torch.float32))
        return torch.mean(einops.reduce(torch.abs(gradient - sample_noise), "batch sample features -> batch sample", "sum"))

    def inference(self, sample_size:int, device:torch.device) -> torch.Tensor:

        x_t = torch.randn((1, sample_size, self.img_width * self.img_height * self.img_channel), device = device)

        for T in range(self.time_step - 1, 0, -1):
            alpha = self.alpha_buffer[T]
            beta = 1 - alpha
            alpha_cum = self.alpha_cumsum_buffer[T]

            timestamp = torch.tensor([T], device=device, dtype = torch.float32)
            timestamp = einops.repeat(timestamp, "one -> batch sample one", batch = 1, sample = sample_size)
            noise = torch.randn((sample_size, self.img_width * self.img_height * self.img_channel)).to(device)
            x_t = alpha**-0.5 * (x_t - (1 - alpha) * (1 - alpha_cum)**-0.5 * self.error_gradient.forward(x_t, timestamp)) + beta**0.5 * noise

        return einops.rearrange(
            x_t,
            "batch sample (c h w) -> (batch sample) c h w",
            sample=sample_size,
            c=self.img_channel,
            h=self.img_height,
            w=self.img_width,
        )