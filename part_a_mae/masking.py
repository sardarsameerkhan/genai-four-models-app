import torch


def random_masking(x, mask_ratio=0.75):
    """
    Per-sample random masking by sorting uniform noise.
    Args:
        x: (N, L, D) patch embeddings (here L=64, D=192)
        mask_ratio: fraction of patches to mask (0.75 -> keep 16, mask 48)
    Returns:
        x_visible: (N, len_keep, D) visible patch tokens
        mask: (N, L) binary mask where 1 = masked, 0 = visible
        ids_restore: (N, L) indices to restore original patch order
    """
    N, L, D = x.shape
    len_keep = int(L * (1.0 - mask_ratio))

    # Generate random noise per sample and sort to get random permutations
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first `len_keep` patches (16 out of 64)
    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Build binary mask: 0 is visible, 1 is masked (used for loss & visualization)
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_visible, mask, ids_restore


if __name__ == "__main__":
    # Correctness check (Section 3.6): verify 48 of 64 patches are masked in every image
    dummy_patches = torch.randn(8, 64, 192)
    visible, mask, ids_restore = random_masking(dummy_patches, mask_ratio=0.75)

    assert visible.shape == (8, 16, 192), f"Expected (8, 16, 192), got {visible.shape}"
    masked_counts = mask.sum(dim=1)
    assert torch.all(masked_counts == 48), f"Expected 48 masked patches per image, got {masked_counts}"

    print("MAE Masking Check Passed!")
    print(f"Visible shape: {tuple(visible.shape)} | Masked patches per image: {masked_counts.int().tolist()}")