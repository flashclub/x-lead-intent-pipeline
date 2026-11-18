# 发布步骤（token 就绪后）

```bash
cd /Users/deepmac/Documents/code/x-lead-intent-pipeline/huggingface
export HF_TOKEN=hf_xxx   # write 权限, https://huggingface.co/settings/tokens
../monoboard-venv/python publish_hf.py   # 或任何装有 huggingface_hub 的 python
```

脚本会自动：
1. 创建 model 仓库 `{user}/tweet-buy-intent` 并上传 head.pt + metrics.json + README（模型卡）
2. 创建 Space `{user}/tweet-buy-intent-space`（gradio demo，上传 app.py + README + requirements）
3. Space 首次启动会下载 embeddinggemma-2（~1.5GB），CPU 免费档即可运行
