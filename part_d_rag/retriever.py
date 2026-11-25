import sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bioasq import load_corpus, load_questions


class DenseRetriever:
    """
    Dense passage retriever built from scratch on top of sentence-transformers
    (all-MiniLM-L6-v2) using L2-normalized embeddings and exact cosine similarity.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.cache_dir = Path(cache_dir) if cache_dir else (ROOT / "part_d_rag")
        self.doc_ids = []
        self.doc_texts = []
        self.embeddings = None  # (num_docs, dim) float32 L2-normalized

    def build_index(self, corpus_dict, use_cache=True):
        """
        Encodes all passages in corpus_dict ({int_id: text}) into normalized vectors.
        """
        self.doc_ids = list(corpus_dict.keys())
        self.doc_texts = [corpus_dict[did] for did in self.doc_ids]

        emb_path = self.cache_dir / "corpus_embeddings.npy"
        ids_path = self.cache_dir / "corpus_doc_ids.npy"

        if use_cache and emb_path.exists() and ids_path.exists():
            cached_ids = np.load(ids_path).tolist()
            if cached_ids == self.doc_ids:
                self.embeddings = np.load(emb_path)
                return

        self.embeddings = self.encoder.encode(
            self.doc_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, self.embeddings)
        np.save(ids_path, np.array(self.doc_ids, dtype=np.int64))

    def retrieve(self, query, k=5):
        """
        Returns top-k passages as a list of dicts:
        [{"doc_id": int, "score": float, "text": str}, ...]
        """
        q_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        scores = self.embeddings @ q_emb  # exact cosine similarity
        top_idx = np.argsort(-scores)[:k]
        return [
            {
                "doc_id": self.doc_ids[i],
                "score": float(scores[i]),
                "text": self.doc_texts[i],
            }
            for i in top_idx
        ]

    def evaluate_recall_at_k(self, qa_list, ks=(1, 3, 5, 10)):
        """
        Computes Recall@k: fraction of questions where at least one gold relevant
        doc_id appears in the top-k retrieved passages.
        """
        max_k = max(ks)
        queries = [item["question"] for item in qa_list]
        q_embs = self.encoder.encode(
            queries,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        sim_matrix = q_embs @ self.embeddings.T  # (num_queries, num_docs)
        top_indices = np.argpartition(-sim_matrix, kth=max_k - 1, axis=1)[:, :max_k]

        hits = {k: 0 for k in ks}
        valid = 0
        for row_idx, item in enumerate(qa_list):
            gold_ids = set(item.get("relevant_ids", []))
            if not gold_ids:
                continue
            valid += 1
            cand_idx = top_indices[row_idx]
            ordered_idx = cand_idx[np.argsort(-sim_matrix[row_idx, cand_idx])]
            ranked_ids = [self.doc_ids[i] for i in ordered_idx]

            for k in ks:
                if any(did in gold_ids for did in ranked_ids[:k]):
                    hits[k] += 1

        return {k: (hits[k] / max(valid, 1)) * 100.0 for k in ks}


if __name__ == "__main__":
    corpus = load_corpus()
    test_qa = load_questions(split="test", corpus=corpus)
    print(f"Loaded BioASQ: {len(corpus):,} passages | {len(test_qa)} test Qs")

    retriever = DenseRetriever()
    retriever.build_index(corpus, use_cache=True)
    recalls = retriever.evaluate_recall_at_k(test_qa, ks=(1, 3, 5, 10))
    for k, val in recalls.items():
        print(f"Recall@{k:<2d}: {val:.2f}%")