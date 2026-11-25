import sys
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eurosat import load_eurosat_split, CLASS_NAMES, patchify, unpatchify
from uhat import load_uhat_characters
from bioasq import load_corpus

from part_a_mae.model import MaskedAutoencoder, LinearProbeClassifier
from part_b_vae.model import ConvVAE
from part_c_diffusion.ddpm import ClassConditionalDDPM
from part_d_rag.retriever import DenseRetriever
from part_d_rag.generator import FlanT5Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy-loaded model & dataset cache so startup is instant
CACHE = {}


def tensor_to_pil(t_img, size=192):
    """Converts (C, H, W) tensor in [0, 1] to crisp nearest-neighbor upscaled PIL Image."""
    arr = (t_img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode="L")
    else:
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    return pil.resize((size, size), Image.NEAREST)


def make_strip_pil(imgs, nrow=8, cell_size=96, pad=4):
    """Arranges (N, C, H, W) tensor in [0, 1] into a clean PIL grid."""
    N = imgs.size(0)
    ncol = min(nrow, N)
    nrows = (N + ncol - 1) // ncol
    mode = "L" if imgs.size(1) == 1 else "RGB"
    bg = 240 if mode == "L" else (240, 240, 240)
    canvas = Image.new(mode, (ncol * (cell_size + pad) + pad, nrows * (cell_size + pad) + pad), color=bg)
    for idx in range(N):
        r, c = divmod(idx, ncol)
        tile = tensor_to_pil(imgs[idx], size=cell_size)
        canvas.paste(tile, (pad + c * (cell_size + pad), pad + r * (cell_size + pad)))
    return canvas


# -------------------- PART A: MAE --------------------
def get_mae_bundle():
    if "mae" not in CACHE:
        x_te, y_te = load_eurosat_split("test")
        mae = MaskedAutoencoder().to(DEVICE)
        for ckpt_name in ("mae_pretrained.pt", "mae_eurosat.pt"):
            mae_ckpt = ROOT / "part_a_mae" / ckpt_name
            if mae_ckpt.exists():
                mae.load_state_dict(torch.load(mae_ckpt, map_location=DEVICE))
                break
        mae.eval()

        clf = LinearProbeClassifier(in_dim=192, num_classes=len(CLASS_NAMES)).to(DEVICE)
        for clf_name in ("mae_linear_probe.pt", "mae_classifier.pt", "linear_probe.pt"):
            clf_ckpt = ROOT / "part_a_mae" / clf_name
            if clf_ckpt.exists():
                clf.load_state_dict(torch.load(clf_ckpt, map_location=DEVICE))
                break
        clf.eval()
        CACHE["mae"] = (x_te, y_te, mae, clf)
    return CACHE["mae"]


@torch.no_grad()
def run_mae_demo(sample_idx, mask_ratio, uploaded_img):
    x_te, y_te, mae, clf = get_mae_bundle()
    if uploaded_img is not None:
        pil = uploaded_img.convert("RGB").resize((64, 64))
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        true_label = "Custom Upload"
    else:
        idx = int(sample_idx) % len(x_te)
        x = x_te[idx]
        true_label = CLASS_NAMES[int(y_te[idx])]

    x_batch = x.unsqueeze(0).to(DEVICE)
    mae.mask_ratio = float(mask_ratio)
    torch.manual_seed(42)
    loss, pred_patches, mask = mae(x_batch)

    patches = patchify(x_batch)
    masked_patches = patches * (1.0 - mask.unsqueeze(-1))
    masked_img = unpatchify(masked_patches)[0]
    recon_img = unpatchify(pred_patches)[0]

    feats = mae.extract_features(x_batch)
    logits = clf(feats)
    probs = torch.softmax(logits, dim=-1)[0]
    top3_p, top3_i = torch.topk(probs, k=3)
    pred_summary = (
        f"Ground Truth: {true_label}\n"
        f"Masked Patch MSE: {loss.item():.5f}\n"
        f"Top-3 Classifier Predictions:\n"
        + "\n".join([f"  {i+1}. {CLASS_NAMES[c.item()]} ({p.item()*100:.1f}%)" for i, (p, c) in enumerate(zip(top3_p, top3_i))])
    )
    return tensor_to_pil(x), tensor_to_pil(masked_img), tensor_to_pil(recon_img), pred_summary


# -------------------- PART B: VAE --------------------
def get_vae_bundle():
    if "vae" not in CACHE:
        uhat_root = str(ROOT / "starter" / "data" / "uhat")
        x_te, y_te, names = load_uhat_characters(root=uhat_root, split="test")
        vae = ConvVAE(latent_dim=16).to(DEVICE)
        ckpt = ROOT / "part_b_vae" / "vae_uhat.pt"
        if ckpt.exists():
            vae.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        vae.eval()
        CACHE["vae"] = (x_te, y_te, names, vae)
    return CACHE["vae"]


@torch.no_grad()
def run_vae_prior(num_samples, seed):
    _, _, _, vae = get_vae_bundle()
    torch.manual_seed(int(seed))
    z = torch.randn(int(num_samples), 16, device=DEVICE)
    imgs = vae.decode(z)
    return make_strip_pil(imgs, nrow=8, cell_size=72)


@torch.no_grad()
def run_vae_interp(class_a, class_b):
    x_te, y_te, names, vae = get_vae_bundle()
    idx_a = (y_te == names.index(class_a)).nonzero(as_tuple=True)[0][0]
    idx_b = (y_te == names.index(class_b)).nonzero(as_tuple=True)[0][0]
    mu_a, _ = vae.encode(x_te[idx_a : idx_a + 1].to(DEVICE))
    mu_b, _ = vae.encode(x_te[idx_b : idx_b + 1].to(DEVICE))
    alphas = torch.linspace(0.0, 1.0, steps=10, device=DEVICE)
    zs = torch.cat([(1.0 - a) * mu_a + a * mu_b for a in alphas], dim=0)
    imgs = vae.decode(zs)
    return make_strip_pil(imgs, nrow=10, cell_size=72)


# -------------------- PART C: DDPM --------------------
def get_ddpm_bundle():
    if "ddpm" not in CACHE:
        uhat_root = str(ROOT / "starter" / "data" / "uhat")
        _, _, names = load_uhat_characters(root=uhat_root, split="test")
        ddpm = ClassConditionalDDPM(num_classes=len(names), num_timesteps=1000).to(DEVICE)
        ckpt = ROOT / "part_c_diffusion" / "ddpm_uhat.pt"
        if ckpt.exists():
            ddpm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        ddpm.eval()
        CACHE["ddpm"] = (names, ddpm)
    return CACHE["ddpm"]


@torch.no_grad()
def run_ddpm_demo(class_name, num_samples, seed):
    names, ddpm = get_ddpm_bundle()
    cls_id = names.index(class_name)
    torch.manual_seed(int(seed))
    labels = torch.full((int(num_samples),), cls_id, device=DEVICE, dtype=torch.long)
    capture = [1000, 750, 500, 250, 100, 50, 0]
    imgs, snaps = ddpm.sample(labels, capture_steps=capture)
    traj_imgs = torch.stack([snaps[s][0] for s in capture], dim=0)
    return make_strip_pil(imgs, nrow=4, cell_size=96), make_strip_pil(traj_imgs, nrow=7, cell_size=76)


# -------------------- PART D: RAG --------------------
def get_rag_bundle():
    if "rag" not in CACHE:
        corpus = load_corpus()
        retriever = DenseRetriever()
        retriever.build_index(corpus, use_cache=True)
        generator = FlanT5Generator(model_name="google/flan-t5-base")
        CACHE["rag"] = (retriever, generator)
    return CACHE["rag"]


def run_rag_demo(question, top_k):
    retriever, generator = get_rag_bundle()
    docs = retriever.retrieve(question, k=int(top_k))
    cb_ans = generator.answer_single(question, passages=None)
    rag_ans = generator.answer_single(question, passages=docs)
    table_rows = [[i + 1, d["doc_id"], round(d["score"], 4), d["text"][:320] + "..."] for i, d in enumerate(docs)]
    return cb_ans, rag_ans, table_rows


def build_ui():
    uhat_names = [
        "alif", "alifmad", "ain", "baa", "bariyeh", "che", "chotiyeh", "daad",
        "daal", "ddaal", "dzal", "faa", "gaaf", "ghain", "hamza", "haay",
        "jeem", "kaaf", "khay", "laam", "meem", "noon", "noonghunna", "pay",
        "qaaf", "ray", "rray", "saad", "seen", "sheen", "tay", "thay",
        "ttay", "vao", "zaa", "zaal", "zay", "zhay", "dochahay", "ultha_paish"
    ]
    try:
        _, _, loaded_names = load_uhat_characters(root=str(ROOT / "starter" / "data" / "uhat"), split="test")
        uhat_names = loaded_names
    except Exception:
        pass

    with gr.Blocks(title="Four Generative AI Paradigms Explorer") as demo:
        gr.Markdown(
            "# Four Generative AI Paradigms: Interactive Explorer\n"
            "Explore all four trained models end-to-end: **Masked Autoencoder (EuroSAT)**, "
            "**Convolutional VAE (UHaT)**, **Class-Conditional DDPM (UHaT)**, and **Biomedical RAG (BioASQ)**."
        )

        with gr.Tab("Part A: Masked Autoencoder (EuroSAT)"):
            gr.Markdown("Mask random $8\\times 8$ patches of a $64\\times 64$ Sentinel-2 satellite image, reconstruct missing pixels with ViT-MAE, and classify land cover.")
            with gr.Row():
                mae_idx = gr.Slider(0, 5399, value=12, step=1, label="EuroSAT Test Image Index")
                mae_mask = gr.Slider(0.10, 0.90, value=0.75, step=0.05, label="Mask Ratio")
                mae_upload = gr.Image(type="pil", label="Or Upload Custom Satellite Image")
            mae_btn = gr.Button("Reconstruct & Classify", variant="primary")
            with gr.Row():
                mae_orig = gr.Image(label="Original (64x64)")
                mae_masked = gr.Image(label="Masked Input")
                mae_recon = gr.Image(label="MAE Reconstruction")
            mae_out = gr.Textbox(label="Reconstruction MSE & Few-Label Classifier Output", lines=5)
            mae_btn.click(run_mae_demo, [mae_idx, mae_mask, mae_upload], [mae_orig, mae_masked, mae_recon, mae_out])

        with gr.Tab("Part B: Variational Autoencoder (UHaT)"):
            gr.Markdown("Sample handwritten Urdu characters from $z \\sim \\mathcal{N}(0, I)$ or interpolate smoothly in the 16-D latent space.")
            with gr.Row():
                with gr.Column():
                    vae_n = gr.Slider(8, 32, value=16, step=8, label="Number of Prior Samples")
                    vae_seed = gr.Number(value=42, label="Random Seed")
                    vae_sample_btn = gr.Button("Sample from Prior N(0, I)", variant="primary")
                    vae_prior_img = gr.Image(label="Generated Prior Samples")
                with gr.Column():
                    vae_ca = gr.Dropdown(uhat_names, value=uhat_names[0], label="Start Character")
                    vae_cb = gr.Dropdown(uhat_names, value=uhat_names[min(16, len(uhat_names)-1)], label="End Character")
                    vae_interp_btn = gr.Button("Interpolate Latent Space", variant="primary")
                    vae_interp_img = gr.Image(label="10-Step Latent Interpolation")
            vae_sample_btn.click(run_vae_prior, [vae_n, vae_seed], vae_prior_img)
            vae_interp_btn.click(run_vae_interp, [vae_ca, vae_cb], vae_interp_img)

        with gr.Tab("Part C: Class-Conditional Diffusion (DDPM)"):
            gr.Markdown("Generate sharp handwritten Urdu characters conditioned on class label via 1,000-step reverse denoising.")
            with gr.Row():
                ddpm_cls = gr.Dropdown(uhat_names, value=uhat_names[0], label="Target Urdu Character Class")
                ddpm_n = gr.Slider(1, 8, value=4, step=1, label="Samples to Generate")
                ddpm_seed = gr.Number(value=42, label="Random Seed")
            ddpm_btn = gr.Button("Run 1,000-Step Reverse Diffusion", variant="primary")
            with gr.Row():
                ddpm_out = gr.Image(label="Generated Class-Conditional Characters")
                ddpm_traj = gr.Image(label="Denoising Trajectory (t=1000 -> 750 -> 500 -> 250 -> 100 -> 50 -> 0)")
            ddpm_btn.click(run_ddpm_demo, [ddpm_cls, ddpm_n, ddpm_seed], [ddpm_out, ddpm_traj])

        with gr.Tab("Part D: Biomedical RAG (BioASQ)"):
            gr.Markdown("Compare **Closed-Book Flan-T5** vs. **Retrieval-Augmented Generation** over 40,181 PubMed abstracts.")
            rag_q = gr.Textbox(
                value="Glecaprevir and Pibrentasvir are used for treatment of which disease?",
                label="Biomedical Question",
            )
            gr.Examples(
                examples=[
                    ["Glecaprevir and Pibrentasvir are used for treatment of which disease?"],
                    ["Is tubulin acetylation involved in cell motility?"],
                    ["Is ozanezumab effective for amyotrophic lateral sclerosis?"],
                    ["What is the function of the AtxA pleiotropic regulator?"],
                ],
                inputs=rag_q,
            )
            rag_k = gr.Slider(1, 10, value=5, step=1, label="Top-k Retrieved Passages")
            rag_btn = gr.Button("Retrieve & Answer", variant="primary")
            with gr.Row():
                rag_cb = gr.Textbox(label="Closed-Book Answer (No Retrieval)", lines=3)
                rag_ans = gr.Textbox(label="RAG Answer (Grounded in Top-k PubMed Abstracts)", lines=3)
            rag_docs = gr.Dataframe(
                headers=["Rank", "PubMed ID", "Cosine Sim", "Passage Snippet"],
                label="Retrieved Evidence Passages",
                wrap=True,
            )
            rag_btn.click(run_rag_demo, [rag_q, rag_k], [rag_cb, rag_ans, rag_docs])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)