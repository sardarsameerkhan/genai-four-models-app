# Four Generative AI Paradigms: Implementation, Evaluation & Interactive Explorer

An end-to-end PyTorch implementation and unified Gradio web application demonstrating four foundational generative modeling paradigms across remote sensing, handwritten script synthesis, and biomedical question answering:

1. **Part A — Masked Autoencoder (ViT-MAE)** on **EuroSAT** ($64 \times 64$ Sentinel-2 satellite imagery)
2. **Part B — Convolutional Variational Autoencoder (ConvVAE)** on **UHaT** ($28 \times 28$ Urdu handwritten characters)
3. **Part C — Class-Conditional Denoising Diffusion Probabilistic Model (DDPM)** on **UHaT** (40 Urdu character classes)
4. **Part D — Biomedical Retrieval-Augmented Generation (RAG)** on **BioASQ** (40,181 PubMed abstracts & 707 test questions)

---

## Repository Structure

```text
genai-four-models-app/
├── app/
│   └── app.py                      # Unified 4-tab Gradio web interface
├── part_a_mae/
│   ├── masking.py                  # Random patch masking & index restoration
│   ├── model.py                    # Asymmetric ViT-MAE & LinearProbeClassifier
│   ├── pretrain_mae.py             # Self-supervised pretraining (75% mask ratio)
│   └── eval_linear_probe.py        # Downstream land-cover classification
├── part_b_vae/
│   ├── model.py                    # ConvVAE (d_z=16) with reparameterization & ELBO loss
│   └── train_vae.py                # Training, prior sampling, interpolation & reconstruction
├── part_c_diffusion/
│   ├── schedule.py                 # Linear beta noise schedule (T=1000) & forward diffusion
│   ├── unet.py                     # Sinusoidal time + class-conditioned UNet2D
│   ├── ddpm.py                     # Noise prediction training & 1000-step reverse sampling
│   └── train_ddpm.py               # Training & multi-class generation
├── part_d_rag/
│   ├── retriever.py                # Dense cosine-similarity retriever (all-MiniLM-L6-v2)
│   ├── generator.py                # Flan-T5-base closed-book & context-grounded generator
│   └── evaluate_rag.py             # Recall@k, ROUGE-1/2/L & qualitative error analysis
├── results/                        # Training curves, qualitative grids & metric charts
├── starter/                        # Dataset loaders & patch utilities (EuroSAT, UHaT, BioASQ)
├── requirements.txt
└── README.md
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
