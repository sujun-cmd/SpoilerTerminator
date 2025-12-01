import os
import torch
import numpy as np
import joblib
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score

# --- 环境设置 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "vector_models")

# --- 1. 加载模型 ---
print("正在加载保存的分类器...")
clf_path = os.path.join(model_dir, "spoiler_direction_clf.pkl")
if not os.path.exists(clf_path):
    print("错误：找不到模型文件，请先运行 train_vector.py")
    exit()

clf = joblib.load(clf_path)

print("正在加载句向量模型...")
embed_model = SentenceTransformer('all-mpnet-base-v2')
if torch.cuda.is_available():
    embed_model = embed_model.to('cuda')

# --- 2. 准备测试数据 (关键修改) ---
print("正在准备测试集...")
raw_dataset = load_dataset("bhavyagiri/imdb-spoiler")

# [关键修改] 使用和训练集一模一样的切分方式
split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
test_data = split["test"]  # 拿出另外 20% 作为测试集

print(f"测试集文档数: {len(test_data)}")

test_sentences = []
test_labels = []

for text, doc_label in zip(test_data["text"], test_data["label"]):
    sents = sent_tokenize(text)
    for sent in sents:
        if len(sent) < 10: continue
        test_sentences.append(sent)
        test_labels.append(doc_label)

# --- 3. 预测与评估 ---
print(f"正在计算 {len(test_sentences)} 个句子的向量...")
X_test = embed_model.encode(test_sentences, batch_size=64, show_progress_bar=True)

print("正在推理...")
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

# --- 4. 输出指标 ---
print("\n" + "="*40)
print("Evaluation Report (Sentence Level)")
print("="*40)

# 使用 0.5 阈值
print(classification_report(test_labels, y_pred, target_names=["Safe", "Spoiler"]))

f1 = f1_score(test_labels, y_pred)
print(f"Final F1 Score: {f1:.4f}")

# --- 5. 精彩时刻 ---
print("\n" + "="*40)
print("模型认为最剧透的句子 (Top Spoilers Detected)")
print("="*40)

top_indices = np.argsort(y_probs)[-10:][::-1]

for idx in top_indices:
    sent = test_sentences[idx]
    score = y_probs[idx]
    # 注意：这里的 labels 是弱标签（文档标签），仅供参考
    true_label = "Spoiler Doc" if test_labels[idx] == 1 else "Safe Doc"
    print(f"Score: {score:.4f} | [{true_label}] {sent}")

print("\n" + "="*40)
print("自定义测试用例")
print("="*40)
custom_sents = [
    "The camera work is fantastic.",
    "Bruce Willis is a ghost.",
    "He dies at the end.",
    "It was a boring movie."
]
custom_vecs = embed_model.encode(custom_sents)
custom_probs = clf.predict_proba(custom_vecs)[:, 1]

for sent, prob in zip(custom_sents, custom_probs):
    label = "🚨 SPOILER" if prob > 0.5 else "✅ SAFE"
    print(f"[{prob:.4f}] {label} : {sent}")
