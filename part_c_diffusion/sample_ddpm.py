import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from uhat import load_uhat_characters
from viz import save_grid
from part_c_diffusion.ddpm import ClassConditionalDDPM


def measure_sharpness_and_diversity(imgs):
    """
    Computes:
    - Sharpness: mean absolute spatial gradient (higher = crisper ink edges, less blur)
    - Diversity: mean pairwise L2 distance across generated images (higher = more variety)
    """
    dx = (imgs[:, :, :, 1:] - imgs[:, :, :, :-1]).abs().mean()
    dy = (imgs[:, :, 1:, :] - imgs[:, :, :-1, :]).abs().mean()
    sharpness = ((dx + dy) * 0.5).item()

    flat = imgs.view(imgs.size(0), -1)
    dists = torch.pdist(flat, p=2)
    diversity = dists.mean().item()
    return sharpness, diversity


@torch.no_grad()
def sample_and_evaluate_ddpm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sampling DDPM on device: {device}")

    uhat_root = str(ROOT / "starter" / "data" / "uhat")
    x_te, _, names = load_uhat_characters(root=uhat_root, split="test")

    model = ClassConditionalDDPM(num_classes=len(names), num_timesteps=1000).to(device)
    model.load_state_dict(torch.load(ROOT / "part_c_diffusion" / "ddpm_uhat.pt", map_location=device))
    model.eval()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    torch.manual_seed(42)

    # 1. Generate 4 samples per class for all 40 classes (160 images -> 20 rows x 8 cols)
    t0 = time.time()
    all_labels = torch.arange(len(names), device=device).repeat_interleave(4)  # [0,0,0,0, 1,1,1,1, ...]
    samples_160 = model.sample(all_labels).cpu()
    sample_time = time.time() - t0
    print(f"Generated 160 class-conditional samples (1,000 steps) in {sample_time:.2f}s")

    titles_160 = [names[lbl.item()] for lbl in all_labels.cpu()]
    save_grid(samples_160, results_dir / "ddpm_40class_grid.png", nrow=8, titles=titles_160)

    # 2. Generate 8-step reverse denoising trajectory for 4 distinct Urdu letters
    traj_classes = [0, 3, 16, 21]  # e.g., alif, baa, jeem, noon
    traj_labels = torch.tensor(traj_classes, device=device)
    capture_steps = [1000, 800, 600, 400, 200, 100, 50, 0]
    _, snapshots = model.sample(traj_labels, capture_steps=capture_steps)

    # Arrange as 4 rows (one per letter) x 8 columns (t=1000 -> t=0)
    traj_tiles = []
    traj_titles = []
    for row_idx, cls_id in enumerate(traj_classes):
        for step in capture_steps:
            traj_tiles.append(snapshots[step][row_idx])
            traj_titles.append(f"{names[cls_id]} t={step}")
    traj_tensor = torch.stack(traj_tiles, dim=0)
    save_grid(traj_tensor, results_dir / "ddpm_denoising_trajectory.png", nrow=8, titles=traj_titles)

    # 3. Sharpness & Diversity metrics for Analysis E.3 (Real vs DDPM)
    real_sharp, real_div = measure_sharpness_and_diversity(x_te[:160])
    ddpm_sharp, ddpm_div = measure_sharpness_and_diversity(samples_160)
    print(f"Real UHaT (160) -> Edge Sharpness: {real_sharp:.4f} | Pairwise Diversity: {real_div:.2f}")
    print(f"DDPM (160)      -> Edge Sharpness: {ddpm_sharp:.4f} | Pairwise Diversity: {ddpm_div:.2f}")

    # Also evaluate VAE if checkpoint exists locally
    vae_ckpt = ROOT / "part_b_vae" / "vae_uhat.pt"
    if vae_ckpt.exists():
        from part_b_vae.model import ConvVAE
        vae = ConvVAE(latent_dim=16).to(device)
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        vae.eval()
        vae_samples = vae.decode(torch.randn(160, 16, device=device)).cpu()
        vae_sharp, vae_div = measure_sharpness_and_diversity(vae_samples)
        print(f"VAE (160)       -> Edge Sharpness: {vae_sharp:.4f} | Pairwise Diversity: {vae_div:.2f}")

    print("Saved results/ddpm_40class_grid.png and results/ddpm_denoising_trajectory.png")


if __name__ == "__main__":
    sample_and_evaluate_ddpm()