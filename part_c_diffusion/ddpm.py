import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class LinearNoiseSchedule:
    """
    Linear beta schedule from beta_1 = 1e-4 to beta_T = 0.02 over T = 1000 steps.
    Precomputes alpha_t and cumulative product alpha_bar_t on the target device.
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(self, x0, t, noise=None):
        """
        Closed-form forward diffusion q(x_t | x_0):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        Args:
            x0: (N, 1, 28, 28) clean images scaled to [-1, 1]
            t:  (N,) integer timesteps in [0, T - 1]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise


class ClassConditionalDDPM(nn.Module):
    """
    Class-conditional DDPM using diffusers' UNet2DModel as the noise predictor
    and a custom PyTorch linear schedule + 1000-step reverse sampling loop.
    """
    def __init__(self, num_classes=40, num_timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.schedule = LinearNoiseSchedule(num_timesteps=num_timesteps)

        self.unet = UNet2DModel(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            num_class_embeds=num_classes,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.schedule.to(device)
        return self

    def forward(self, x0, labels):
        """
        Training step: sample random t in [0, T-1], add closed-form Gaussian noise,
        predict noise with UNet2DModel, and compute MSE ||eps - eps_theta||^2.
        """
        device = x0.device
        N = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (N,), device=device, dtype=torch.long)
        x_t, noise = self.schedule.q_sample(x0, t)
        pred_noise = self.unet(x_t, t, class_labels=labels).sample
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, labels, capture_steps=None):
        """
        Full T-step DDPM reverse sampling loop starting from x_T ~ N(0, I):
        x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta) + sqrt(beta_t) * z
        Returns images rescaled back to [0, 1] (and optional intermediate snapshots).
        """
        self.eval()
        device = next(self.parameters()).device
        labels = labels.to(device)
        N = labels.size(0)

        x = torch.randn(N, 1, 28, 28, device=device)
        snapshots = {}
        if capture_steps and self.num_timesteps in capture_steps:
            snapshots[self.num_timesteps] = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()

        for step in reversed(range(self.num_timesteps)):
            t_batch = torch.full((N,), step, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch, class_labels=labels).sample

            beta_t = self.schedule.betas[step]
            alpha_t = self.schedule.alphas[step]
            alpha_bar_t = self.schedule.alpha_bars[step]

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * pred_noise)

            if step > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * z
            else:
                x = mean

            human_step = step  # step=0 corresponds to clean x_0
            if capture_steps and human_step in capture_steps:
                snapshots[human_step] = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()

        imgs_01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        if capture_steps is not None:
            return imgs_01, snapshots
        return imgs_01


if __name__ == "__main__":
    # 1. Forward schedule check (Section 3.6): verify x_T (t=999) has mean ~ 0 and std ~ 1
    schedule = LinearNoiseSchedule(num_timesteps=1000)
    clean_imgs = torch.empty(256, 1, 28, 28).uniform_(-1.0, 1.0)
    t_final = torch.full((256,), 999, dtype=torch.long)
    x_T, _ = schedule.q_sample(clean_imgs, t_final)

    mean_T = x_T.mean().item()
    std_T = x_T.std().item()
    assert abs(mean_T) < 0.05, f"Expected x_T mean near 0, got {mean_T}"
    assert abs(std_T - 1.0) < 0.05, f"Expected x_T std near 1, got {std_T}"

    # 2. Parameter count check for Table 2
    ddpm = ClassConditionalDDPM(num_classes=40, num_timesteps=1000)
    total_params = sum(p.numel() for p in ddpm.parameters())
    print(f"DDPM Total Parameters (UNet2DModel): {total_params:,}")
    print(f"DDPM Forward Noise Check Passed! At t=1000: mean={mean_T:.4f}, std={std_T:.4f}")