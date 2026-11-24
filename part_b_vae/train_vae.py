import sys
import time
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from uhat import load_uhat_characters
from part_b_vae.model import ConvVAE, vae_loss


def plot_vae_curves(totals, bces, klds, save_path):
    epochs = range(1, len(totals) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].plot(epochs, totals, linewidth=2, color="tab:blue")
    axes[0].set_title("Total Negative ELBO Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, bces, linewidth=2, color="tab:green")
    axes[1].set_title("Reconstruction BCE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, klds, linewidth=2, color="tab:orange")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train_vae(epochs=30, batch_size=128, lr=1e-3, latent_dim=16, beta=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ConvVAE (d_z={latent_dim}, beta={beta}) on device: {device}")

    uhat_root = str(ROOT / "starter" / "data" / "uhat")
    x_tr, _, names = load_uhat_characters(root=uhat_root, split="train")
    train_loader = DataLoader(TensorDataset(x_tr), batch_size=batch_size, shuffle=True)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist_total, hist_bce, hist_kld = [], [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        sum_total, sum_bce, sum_kld = 0.0, 0.0, 0.0

        for (batch_imgs,) in train_loader:
            batch_imgs = batch_imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_imgs)
            loss, bce, kld = vae_loss(recon, batch_imgs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            n = batch_imgs.size(0)
            sum_total += loss.item() * n
            sum_bce += bce.item() * n
            sum_kld += kld.item() * n

        n_samples = len(x_tr)
        avg_total = sum_total / n_samples
        avg_bce = sum_bce / n_samples
        avg_kld = sum_kld / n_samples

        hist_total.append(avg_total)
        hist_bce.append(avg_bce)
        hist_kld.append(avg_kld)

        if epoch == 1 or epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:2d}/{epochs}] | Total: {avg_total:.2f} | "
                f"BCE: {avg_bce:.2f} | KL: {avg_kld:.2f} | Elapsed: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print(
        f"Finished VAE training in {total_time:.1f}s | "
        f"Final Total: {hist_total[-1]:.2f} | Final BCE: {hist_bce[-1]:.2f} | Final KL: {hist_kld[-1]:.2f}"
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    ckpt_path = ROOT / "part_b_vae" / "vae_uhat.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights to {ckpt_path}")

    plot_vae_curves(hist_total, hist_bce, hist_kld, results_dir / "vae_training_curves.png")
    print("Saved results/vae_training_curves.png")


if __name__ == "__main__":
    train_vae()