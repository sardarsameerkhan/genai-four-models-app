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
from part_c_diffusion.ddpm import ClassConditionalDDPM


def plot_ddpm_loss(losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2, color="tab:purple")
    plt.xlabel("Epoch")
    plt.ylabel("Noise Prediction MSE Loss")
    plt.title("Class-Conditional DDPM Training Loss (UHaT)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_ddpm(epochs=25, batch_size=128, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ClassConditionalDDPM on device: {device}")

    uhat_root = str(ROOT / "starter" / "data" / "uhat")
    x_tr, y_tr, names = load_uhat_characters(root=uhat_root, split="train")
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = ClassConditionalDDPM(num_classes=len(names), num_timesteps=1000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_imgs, batch_labels in train_loader:
            # Rescale from [0, 1] to [-1, 1] as required by Section C.2
            x0 = batch_imgs.to(device) * 2.0 - 1.0
            labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = model(x0, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x0.size(0)

        epoch_loss = total_loss / len(x_tr)
        losses.append(epoch_loss)
        if epoch == 1 or epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:2d}/{epochs}] | Noise MSE: {epoch_loss:.5f} | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"Finished DDPM training in {total_time:.1f}s | Final Noise MSE: {losses[-1]:.5f}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    ckpt_path = ROOT / "part_c_diffusion" / "ddpm_uhat.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights to {ckpt_path}")

    plot_ddpm_loss(losses, results_dir / "ddpm_train_loss.png")
    print("Saved results/ddpm_train_loss.png")


if __name__ == "__main__":
    train_ddpm()