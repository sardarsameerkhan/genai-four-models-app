import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for UHaT (28 x 28 grayscale).
    Encoder: 28x28 -> 14x14 (32 ch) -> 7x7 (64 ch) -> 4x4 (128 ch) -> mu, logvar (d_z)
    Decoder: d_z -> 128x7x7 -> 64x14x14 -> 1x28x28 (Sigmoid)
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 7 -> 4
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, 128 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """z = mu + exp(0.5 * logvar) * eps, with eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, 128, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Negative ELBO averaged per sample:
    Loss = Reconstruction BCE + beta * KL(q(z|x) || N(0, I))
    """
    batch_size = x.size(0)
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total = bce + beta * kld
    return total, bce, kld


if __name__ == "__main__":
    model = ConvVAE(latent_dim=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE Total Parameters (d_z=16): {total_params:,}")

    # Correctness checks (Section 3.6):
    # 1. KL divergence of standard normal (mu=0, logvar=0) must equal 0.0
    zero_mu = torch.zeros(8, 16)
    zero_logvar = torch.zeros(8, 16)
    dummy_img = torch.rand(8, 1, 28, 28)
    _, _, kld_zero = vae_loss(dummy_img, dummy_img, zero_mu, zero_logvar, beta=1.0)
    assert torch.isclose(kld_zero, torch.tensor(0.0), atol=1e-6), f"Expected KL=0, got {kld_zero}"

    # 2. Forward pass & gradient flow through reparameterization trick
    recon, mu, logvar = model(dummy_img)
    assert recon.shape == (8, 1, 28, 28)
    loss, bce, kld = vae_loss(recon, dummy_img, mu, logvar, beta=1.0)
    loss.backward()
    assert model.fc_mu.weight.grad is not None, "Gradients must flow back through reparameterization!"

    print(f"VAE Model, KL=0 & Reparameterization Gradient Checks Passed! (BCE={bce.item():.2f}, KL={kld.item():.4f})")