import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from uhat import load_uhat_characters
from viz import save_grid
from part_b_vae.model import ConvVAE, vae_loss


@torch.no_grad()
def evaluate_vae(latent_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uhat_root = str(ROOT / "starter" / "data" / "uhat")

    _, _, names = load_uhat_characters(root=uhat_root, split="train")
    x_te, y_te, _ = load_uhat_characters(root=uhat_root, split="test", class_names=names)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(ROOT / "part_b_vae" / "vae_uhat.pt", map_location=device))
    model.eval()

    # Compute overall test set ELBO, BCE, and KL
    x_te_dev = x_te.to(device)
    recon_all, mu_all, logvar_all = model(x_te_dev)
    test_total, test_bce, test_kld = vae_loss(recon_all, x_te_dev, mu_all, logvar_all, beta=1.0)
    print(f"Test Set Metrics (4,747 images) -> Total: {test_total.item():.2f} | BCE: {test_bce.item():.2f} | KL: {test_kld.item():.2f}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    torch.manual_seed(42)

    # 1. Reconstructions: 16 test characters side-by-side (original, reconstruction)
    idx_16 = torch.randperm(len(x_te))[:16]
    orig_16 = x_te_dev[idx_16]
    recon_16, _, _ = model(orig_16)
    # Interleave [orig_0, recon_0, orig_1, recon_1, ...] in an 8-column grid (4 rows x 8 cols = 32 tiles)
    pairs = torch.stack([orig_16, recon_16], dim=1).reshape(32, 1, 28, 28)
    pair_titles = []
    for idx in idx_16:
        lbl = names[y_te[idx].item()]
        pair_titles.extend([f"{lbl} (orig)", f"{lbl} (recon)"])
    save_grid(pairs, results_dir / "vae_reconstructions.png", nrow=8, titles=pair_titles)

    # 2. Prior Samples: 64 samples drawn from z ~ N(0, I)
    z_prior = torch.randn(64, latent_dim, device=device)
    samples_64 = model.decode(z_prior)
    save_grid(samples_64, results_dir / "vae_prior_samples.png", nrow=8)

    # 3. Latent Interpolation: 8 pairs of test letters, 10 steps each (8 rows x 10 cols = 80 tiles)
    idx_a = torch.randperm(len(x_te))[:8]
    idx_b = torch.randperm(len(x_te))[8:16]
    mu_a, _ = model.encode(x_te_dev[idx_a])
    mu_b, _ = model.encode(x_te_dev[idx_b])

    alphas = torch.linspace(0.0, 1.0, steps=10, device=device)
    interp_imgs = []
    for i in range(8):
        z_line = torch.stack([(1.0 - a) * mu_a[i] + a * mu_b[i] for a in alphas], dim=0)
        interp_imgs.append(model.decode(z_line))
    interp_grid = torch.cat(interp_imgs, dim=0)  # (80, 1, 28, 28)
    save_grid(interp_grid, results_dir / "vae_interpolations.png", nrow=10)

    print("Saved results/vae_reconstructions.png, results/vae_prior_samples.png, and results/vae_interpolations.png")


if __name__ == "__main__":
    evaluate_vae()