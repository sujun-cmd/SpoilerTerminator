from transformers import pipeline
import torch

# 加载你刚才训练完的 Self-Training 模型
model_path = "./spoiler_roberta_final"
print(f"正在加载模型: {model_path} ...")

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device)

# 测试用例：包含简单剧透、隐晦剧透、安全句、以及之前被标错的句子
test_sentences = [
    "The camera work is fantastic.",                     # 安全
    "He dies at the end.",                               # 简单剧透 (关键词)
    "Bruce Willis is a ghost.",                          # 隐晦剧透 (语义)
    "It turns out she was the killer all along.",        # 强语义剧透
    "Why didn't Obi-Wan kill Anakin?",                   # 之前被原始数据标错的句子
    "Selina chopping up her mate.",                      # 之前被原始数据标错的句子
    "The plot was boring."                               # 安全（虽然有 plot 关键词）
]

print("-" * 50)
results = classifier(test_sentences)

for text, res in zip(test_sentences, results):
    label = "🚨 SPOILER" if res['label'] == 'LABEL_1' else "✅ SAFE"
    score = res['score']
    print(f"[{score:.4f}] {label} : {text}")
print("-" * 50)
