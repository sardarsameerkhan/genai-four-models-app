import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eurosat import load_eurosat_split, few_label_indices, CLASS_NAMES
from part_a_mae.model import MaskedAutoencoder, LinearProbeClassifier


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("MAE Few-Label Classifier Confusion Matrix (5,400 Test Images)")

    thresh = cm.max().item() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j].item()
            ax.text(
                j, i, str(val),
                ha="center", va="center", fontsize=7,
                color="white" if val > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def extract_all_features(mae, images, batch_size=256, device="cpu"):
    mae.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        feats.append(mae.extract_features(batch).cpu())
    return torch.cat(feats, dim=0)


def train_and_evaluate_classifier(epochs=200, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running MAE few-label classifier on device: {device}")

    ckpt_path = ROOT / "part_a_mae" / "mae_pretrained.pt"
    mae = MaskedAutoencoder().to(device)
    mae.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Freeze encoder strictly (Section A.3)
    for param in mae.parameters():
        param.requires_grad = False

    x_tr, y_tr = load_eurosat_split("train")
    x_te, y_te = load_eurosat_split("test")

    # Select the fixed 160 few-label training images (16 per class)
    idx_160 = few_label_indices(y_tr, per_class=16, seed=42)
    x_few = x_tr[idx_160]
    y_few = y_tr[idx_160].to(device)

    # Extract frozen 192-D features
    feats_few = extract_all_features(mae, x_few, device=device).to(device)
    feats_test = extract_all_features(mae, x_te, device=device).to(device)

    classifier = LinearProbeClassifier(in_dim=192, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train for 200 epochs with all 160 images in one batch (Section A.3)
    for epoch in range(1, epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(feats_few)
        loss = criterion(logits, y_few)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            acc_tr = (logits.argmax(dim=1) == y_few).float().mean().item() * 100
            print(f"Epoch [{epoch:3d}/{epochs}] | Train Loss: {loss.item():.4f} | Train Acc (160): {acc_tr:.2f}%")

    # Evaluate on all 5,400 test images (Section A.4)
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(feats_test)
        preds = test_logits.argmax(dim=1).cpu()

    test_acc = (preds == y_te).float().mean().item() * 100
    print(f"\n>>> MAE Few-Label Test Accuracy (5,400 images): {test_acc:.2f}% (Chance = 10.00%) <<<")

    cm = compute_confusion_matrix(y_te, preds, num_classes=10)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm.numpy())

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    save_confusion_matrix(cm, CLASS_NAMES, results_dir / "mae_confusion_matrix.png")
    torch.save(classifier.state_dict(), ROOT / "part_a_mae" / "mae_classifier.pt")
    print("Saved results/mae_confusion_matrix.png and part_a_mae/mae_classifier.pt")


if __name__ == "__main__":
    train_and_evaluate_classifier()