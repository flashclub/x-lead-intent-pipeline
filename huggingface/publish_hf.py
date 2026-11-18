"""Publish model + Space to Hugging Face. Usage:
    HF_TOKEN=hf_xxx python3 publish_hf.py
Requires: pip install huggingface_hub
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("set HF_TOKEN first (write scope) — https://huggingface.co/settings/tokens")
api = HfApi(token=token)
user = api.whoami()["name"]
model_repo = f"{user}/tweet-buy-intent"
space_repo = f"{user}/tweet-buy-intent-space"
here = Path(__file__).parent

api.create_repo(model_repo, repo_type="model", exist_ok=True, private=False)
api.upload_file(path_or_fileobj=here / "head.pt", path_in_repo="head.pt",
                repo_id=model_repo, repo_type="model")
api.upload_file(path_or_fileobj=here / "metrics.json", path_in_repo="metrics.json",
                repo_id=model_repo, repo_type="model")
api.upload_file(path_or_fileobj=here / "README.md", path_in_repo="README.md",
                repo_id=model_repo, repo_type="model")
print(f"model: https://huggingface.co/{model_repo}")

api.create_repo(space_repo, repo_type="space", exist_ok=True, private=False,
                space_sdk="gradio")
space_dir = here / "space"
api.upload_folder(folder_path=str(space_dir), repo_id=space_repo, repo_type="space")
print(f"space: https://huggingface.co/spaces/{space_repo}")
