import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                               model_kwargs={"dtype": torch.float32})
head = torch.nn.Sequential(
    torch.nn.Linear(768, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))
head.load_state_dict(torch.load(hf_hub_download("flashclub/tweet-buy-intent", "head.pt"),
                                map_location="cpu"))
head.eval()

POSTS = [
    "is there an app that tracks competitor pricing daily? willing to pay",
    "just watched the new Dune movie, cinematography was insane",
]
with torch.no_grad():
    emb = torch.tensor(embedder.encode(POSTS, normalize_embeddings=True))
    scores = torch.sigmoid(head(emb)).squeeze(-1)
for post, s in zip(POSTS, scores.tolist()):
    print(f"{s:.4f}  {post[:60]}")
