import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "google/embeddinggemma-2"
REPO_ID = "flashcoder/tweet-buy-intent"

embedder = SentenceTransformer(EMBED_MODEL, device="cpu",
                               model_kwargs={"dtype": torch.float32})
head = torch.nn.Sequential(
    torch.nn.Linear(768, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))
head.load_state_dict(torch.load(hf_hub_download(REPO_ID, "head.pt"), map_location="cpu"))
head.eval()


def score_post(text: str) -> dict:
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        emb = torch.tensor(embedder.encode([text[:1500]], normalize_embeddings=True))
        score = torch.sigmoid(head(emb)).squeeze(-1).item()
    ms = (time.perf_counter() - t0) * 1000
    tier = "high intent" if score >= 0.8 else "mid intent" if score >= 0.6 else "watchlist" if score >= 0.4 else "noise"
    return {"buy-intent score": round(score, 4), "tier": tier, "latency_ms": round(ms, 1)}


demo = gr.Interface(
    fn=score_post,
    inputs=gr.Textbox(label="X post text", placeholder="is there an app that…"),
    outputs=gr.JSON(label="result"),
    title="Tweet Buy-Intent Miner",
    description="Scores an X/Twitter post 0-1 for buying intent. "
                "Frozen embeddinggemma-2 + a 49k-param MLP head — 100% CPU, single-digit ms. "
                "Pipeline: github.com/flashclub/x-lead-intent-pipeline",
    examples=[
        ["is there an app that tracks competitor pricing daily? willing to pay"],
        ["looking for a cheaper alternative to Ahrefs, $99/mo is killing me"],
        ["just watched the new Dune movie, cinematography was insane"],
        ["can anyone recommend me some cool youtubers to watch?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
