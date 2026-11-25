import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bioasq import load_corpus, load_questions
from part_d_rag.retriever import DenseRetriever
from part_d_rag.generator import FlanT5Generator


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure * 100.0)
        r2.append(scores["rouge2"].fmeasure * 100.0)
        rL.append(scores["rougeL"].fmeasure * 100.0)
    return {
        "ROUGE-1": float(np.mean(r1)),
        "ROUGE-2": float(np.mean(r2)),
        "ROUGE-L": float(np.mean(rL)),
    }


def plot_rag_comparison(recalls, cb_rouge, rag_rouge, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: Retrieval Recall@k
    ks = [f"Recall@{k}" for k in recalls.keys()]
    vals = list(recalls.values())
    bars0 = axes[0].bar(ks, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"], width=0.55)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Dense Retriever Performance (BioASQ Test)")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars0, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Right: Closed-Book vs RAG ROUGE
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    cb_vals = [cb_rouge[m] for m in metrics]
    rag_vals = [rag_rouge[m] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35

    b1 = axes[1].bar(x - w / 2, cb_vals, width=w, label="Closed-Book (No Context)", color="#DD8452")
    b2 = axes[1].bar(x + w / 2, rag_vals, width=w, label="RAG (Top-5 Retrieved)", color="#4C72B0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, max(rag_vals) * 1.25)
    axes[1].set_ylabel("F1 Score (%)")
    axes[1].set_title("Answer Generation Quality: Closed-Book vs. RAG")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    for bar, v in zip(b1, cb_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
    for bar, v in zip(b2, rag_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_rag(top_k=5):
    t0 = time.time()
    corpus = load_corpus()
    test_qa = load_questions(split="test", corpus=corpus)
    print(f"Loaded BioASQ: {len(corpus):,} passages | {len(test_qa)} test Qs")

    retriever = DenseRetriever()
    retriever.build_index(corpus, use_cache=True)

    recalls = retriever.evaluate_recall_at_k(test_qa, ks=(1, 3, 5, 10))
    print("\n--- Retrieval Metrics ---")
    for k, val in recalls.items():
        print(f"Recall@{k:<2d}: {val:.2f}%")

    generator = FlanT5Generator(model_name="google/flan-t5-base")
    questions = [item["question"] for item in test_qa]
    gold_answers = [item["answer"] for item in test_qa]

    # Retrieve top-k passages for each test question
    retrieved_all = [retriever.retrieve(q, k=top_k) for q in questions]

    cb_prompts = [generator.build_closed_book_prompt(q) for q in questions]
    rag_prompts = [generator.build_rag_prompt(q, docs) for q, docs in zip(questions, retrieved_all)]

    print(f"\nGenerating Closed-Book and RAG (k={top_k}) answers on {generator.device}...")
    cb_preds = generator.generate_batch(cb_prompts, max_new_tokens=64, batch_size=32)
    rag_preds = generator.generate_batch(rag_prompts, max_new_tokens=64, batch_size=32)

    cb_rouge = compute_rouge(cb_preds, gold_answers)
    rag_rouge = compute_rouge(rag_preds, gold_answers)

    print("\n--- Generation ROUGE F1 (%) ---")
    print(f"Closed-Book -> ROUGE-1: {cb_rouge['ROUGE-1']:.2f} | ROUGE-2: {cb_rouge['ROUGE-2']:.2f} | ROUGE-L: {cb_rouge['ROUGE-L']:.2f}")
    print(f"RAG (k={top_k})   -> ROUGE-1: {rag_rouge['ROUGE-1']:.2f} | ROUGE-2: {rag_rouge['ROUGE-2']:.2f} | ROUGE-L: {rag_rouge['ROUGE-L']:.2f}")
    print(f"Total RAG Evaluation Time: {time.time() - t0:.1f}s")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    plot_rag_comparison(recalls, cb_rouge, rag_rouge, results_dir / "rag_metrics_comparison.png")
    print("Saved results/rag_metrics_comparison.png")

    # Save 5 qualitative side-by-side examples for report & error analysis
    sample_indices = [0, 5, 12, 25, 40]
    lines = []
    for idx in sample_indices:
        item = test_qa[idx]
        top_doc = retrieved_all[idx][0]
        hit = any(d["doc_id"] in set(item["relevant_ids"]) for d in retrieved_all[idx])
        block = (
            f"=== Example {idx} (Hit@{top_k}={hit}) ===\n"
            f"Question   : {item['question']}\n"
            f"Gold Answer: {item['answer']}\n"
            f"Closed-Book: {cb_preds[idx]}\n"
            f"RAG (k={top_k})  : {rag_preds[idx]}\n"
            f"Top-1 Doc  : [ID={top_doc['doc_id']}, sim={top_doc['score']:.3f}] {top_doc['text'][:240]}...\n"
        )
        lines.append(block)
        print("\n" + block)

    (results_dir / "rag_qualitative_examples.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Saved results/rag_qualitative_examples.txt")


if __name__ == "__main__":
    evaluate_rag(top_k=5)