import os
import torch
import numpy as np
import joblib
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# --- 环境设置 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "vector_models")
os.makedirs(output_dir, exist_ok=True)

# --- 1. 加载与切分数据 (关键修改) ---
print("正在加载数据集...")
raw_dataset = load_dataset("bhavyagiri/imdb-spoiler")

# [关键修改] 手动切分，保留 20% 作为测试集（不参与训练！）
# seed=42 保证每次切分的结果都一样
split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = split["train"]  # 只拿 80% 用来训练

print("正在切分句子并构建训练数据...")
train_sentences = []
train_labels = [] 

for text, doc_label in zip(train_data["text"], train_data["label"]):
    sents = sent_tokenize(text)
    for sent in sents:
        if len(sent) < 10: continue 
        train_sentences.append(sent)
        train_labels.append(doc_label)

print(f"训练集句子总数: {len(train_sentences)}")
print(f"其中正样本 (Label 1): {sum(train_labels)}")

# --- 2. 计算句向量 ---
print("🚀 正在加载句向量模型 (all-mpnet-base-v2)...")
embed_model = SentenceTransformer('all-mpnet-base-v2')
if torch.cuda.is_available():
    embed_model = embed_model.to('cuda')
    print("使用 GPU 加速计算向量")

print("正在计算向量 (Encoding)...")
X_train = embed_model.encode(train_sentences, batch_size=64, show_progress_bar=True)
y_train = np.array(train_labels)

# --- 3. 寻找“剧透方向” ---
print("正在计算剧透特征方向 (Logistic Regression)...")
clf = LogisticRegression(
    random_state=42, 
    solver='liblinear', 
    class_weight='balanced', 
    max_iter=1000,
    C=1.0
)
clf.fit(X_train, y_train)
print("训练完成！")

# --- 4. 保存模型 ---
joblib.dump(clf, os.path.join(output_dir, "spoiler_direction_clf.pkl"))
print(f"模型已保存至: {output_dir}")
