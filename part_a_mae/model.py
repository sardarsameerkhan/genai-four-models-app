import sys
from pathlib import Path
import torch
import torch.nn as nn

# Allow importing from project root and starter/ regardless of working directory
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eurosat import patchify, unpatchify
from part_a_mae.masking import random_masking


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        num_patches=64,
        patch_dim=192,
        enc_dim=192,
        enc_layers=6,
        enc_heads=3,
        enc_ff=768,
        dec_dim=128,
        dec_layers=2,
        dec_heads=4,
        dec_ff=512,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.mask_ratio = mask_ratio

        # --- Patch & Positional Embeddings ---
        self.patch_embed = nn.Linear(patch_dim, enc_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, enc_dim))

        # --- Asymmetric Encoder (sees only 16 visible patches) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=enc_heads,
            dim_feedforward=enc_ff,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.encoder_norm = nn.LayerNorm(enc_dim)

        # --- Lightweight Decoder (reconstructs all 64 patches) ---
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_dim))

        dec_layer = nn.TransformerEncoderLayer(
            d_model=dec_dim,
            nhead=dec_heads,
            dim_feedforward=dec_ff,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.decoder_pred = nn.Linear(dec_dim, patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_encoder(self, patches, mask_ratio=None):
        """
        Embeds patches, adds positional embeddings, masks (if mask_ratio > 0),
        and passes visible tokens through the 6-layer Transformer encoder.
        """
        x = self.patch_embed(patches) + self.pos_embed

        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        if mask_ratio > 0.0:
            x, mask, ids_restore = random_masking(x, mask_ratio=mask_ratio)
        else:
            N, L, _ = x.shape
            mask = torch.zeros(N, L, device=x.device)
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(N, -1)

        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        """
        Projects visible tokens to decoder width (128), inserts learned mask tokens
        at masked positions, restores original order, and predicts patch pixels.
        """
        x = self.decoder_embed(latent)
        N, len_keep, D = x.shape
        num_masked = ids_restore.shape[1] - len_keep

        mask_tokens = self.mask_token.expand(N, num_masked, D)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        x_full = x_full + self.decoder_pos_embed
        x_full = self.decoder(x_full)
        pred = self.decoder_pred(x_full)
        return pred

    def compute_loss(self, target_patches, pred_patches, mask):
        """
        MSE computed strictly on masked patches (mask == 1).
        """
        per_patch_mse = (pred_patches - target_patches).pow(2).mean(dim=-1)  # (N, 64)
        loss = (per_patch_mse * mask).sum() / mask.sum()
        return loss

    def forward(self, images):
        target_patches = patchify(images)  # (N, 64, 192)
        latent, mask, ids_restore = self.forward_encoder(target_patches, mask_ratio=self.mask_ratio)
        pred_patches = self.forward_decoder(latent, ids_restore)
        loss = self.compute_loss(target_patches, pred_patches, mask)
        return loss, pred_patches, mask

    @torch.no_grad()
    def extract_features(self, images):
        """
        For downstream classification (Section A.3): feed all 64 patches (no masking)
        and mean-pool the encoder outputs into a single 192-D vector per image.
        """
        patches = patchify(images)
        latent, _, _ = self.forward_encoder(patches, mask_ratio=0.0)
        return latent.mean(dim=1)  # (N, 192)


class LinearProbeClassifier(nn.Module):
    """Linear classifier on frozen MAE features: nn.Linear(192, 10)."""
    def __init__(self, in_dim=192, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


if __name__ == "__main__":
    model = MaskedAutoencoder()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MAE Total Parameters (encoder + decoder): {total_params:,}")

    # Correctness check (Section 3.6): verify loss ignores visible patches
    imgs = torch.rand(4, 3, 64, 64)
    loss, pred, mask = model(imgs)
    assert mask.sum(dim=1).eq(48).all(), "Mask must have 48 masked patches per image"

    # Perturb predictions at visible positions (mask == 0) and confirm loss is unchanged
    pred_perturbed = pred.clone()
    pred_perturbed[mask == 0] += 999.0
    loss_perturbed = model.compute_loss(patchify(imgs), pred_perturbed, mask)
    assert torch.allclose(loss, loss_perturbed), "Loss must depend ONLY on masked patches!"

    print(f"MAE Model & Masked Loss Check Passed! Initial loss: {loss.item():.4f}")