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

from eurosat import load_eurosat_split, patchify, unpatchify
from viz import save_grid
from part_a_mae.model import MaskedAutoencoder


def save_reconstruction_grid(model, test_images, save_path, num_images=8, device="cpu"):
    """
    Section A.2: Show 3 rows for 8 test images:
    Row 1: Masked input (masked patches grey = 0.5)
    Row 2: Reconstruction (predicted masked patches pasted into visible ones)
    Row 3: Original images
    """
    model.eval()
    imgs = test_images[:num_images].to(device)
    with torch.no_grad():
        _, pred_patches, mask = model(imgs)

    target_patches = patchify(imgs)  # (N, 64, 192)
    mask_expanded = mask.unsqueeze(-1)  # (N, 64, 1)

    # Row 1: masked patches replaced with grey (0.5)
    grey_patches = torch.full_like(target_patches, 0.5)
    masked_input_patches = target_patches * (1.0 - mask_expanded) + grey_patches * mask_expanded
    masked_input_imgs = unpatchify(masked_input_patches)

    # Row 2: visible patches kept, masked patches replaced by model predictions
    recon_patches = target_patches * (1.0 - mask_expanded) + pred_patches.clamp(0.0, 1.0) * mask_expanded
    recon_imgs = unpatchify(recon_patches)

    # Concatenate into 3 rows x 8 columns = 24 images
    grid_imgs = torch.cat([masked_input_imgs, recon_imgs, imgs], dim=0)
    save_grid(grid_imgs, save_path, nrow=num_images)


def plot_loss_curve(losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE Loss")
    plt.title("MAE Pretraining Loss (EuroSAT)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_mae(epochs=100, batch_size=256, lr=5e-4, weight_decay=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MAE on device: {device}")

    x_tr, _ = load_eurosat_split("train")
    x_te, _ = load_eurosat_split("test")

    train_loader = DataLoader(TensorDataset(x_tr), batch_size=batch_size, shuffle=True)

    model = MaskedAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_imgs,) in train_loader:
            batch_imgs = batch_imgs.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(batch_imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_imgs.size(0)

        epoch_loss = total_loss / len(x_tr)
        losses.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:3d}/{epochs}] | Masked MSE: {epoch_loss:.5f} | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"Finished MAE pretraining in {total_time:.1f}s | Final Loss: {losses[-1]:.5f}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    ckpt_path = ROOT / "part_a_mae" / "mae_pretrained.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights to {ckpt_path}")

    plot_loss_curve(losses, results_dir / "mae_pretrain_loss.png")
    save_reconstruction_grid(model, x_te, results_dir / "mae_reconstructions.png", num_images=8, device=device)
    print("Saved results/mae_pretrain_loss.png and results/mae_reconstructions.png")


if __name__ == "__main__":
    train_mae()