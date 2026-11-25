import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class FlanT5Generator:
    """
    Seq2Seq biomedical answer generator using google/flan-t5-base
    for both Closed-Book (parametric only) and RAG (context-grounded) QA.
    """
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def build_closed_book_prompt(question):
        return (
            "Answer the following biomedical question concisely and accurately.\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def build_rag_prompt(question, passages, max_chars_per_doc=650):
        context_blocks = []
        for idx, p in enumerate(passages, start=1):
            snippet = p["text"].strip().replace("\n", " ")[:max_chars_per_doc]
            context_blocks.append(f"[{idx}] {snippet}")
        context_str = "\n".join(context_blocks)
        return (
            "Answer the biomedical question concisely using only the provided PubMed context.\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @torch.no_grad()
    def generate_batch(self, prompts, max_new_tokens=64, batch_size=32):
        outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                early_stopping=True,
            )
            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            outputs.extend([ans.strip() for ans in decoded])
        return outputs

    def answer_single(self, question, passages=None, max_new_tokens=64):
        if passages:
            prompt = self.build_rag_prompt(question, passages)
        else:
            prompt = self.build_closed_book_prompt(question)
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens, batch_size=1)[0]


if __name__ == "__main__":
    gen = FlanT5Generator()
    total_params = sum(p.numel() for p in gen.model.parameters())
    print(f"Flan-T5 Generator Parameters: {total_params:,}")
    sample_q = "What is the mechanism of action of imatinib?"
    print("Closed-Book sample:", gen.answer_single(sample_q))