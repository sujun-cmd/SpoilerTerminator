import os
import time
import torch
import joblib
import numpy as np
import evaluate
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# --- 0. 环境设置 ---
device = 0 if torch.cuda.is_available() else -1
current_dir = os.path.dirname(os.path.abspath(__file__))

# 路径 (根据你实际保存的路径修改)
lr_model_path = os.path.join(current_dir, "vector_models", "spoiler_direction_clf.pkl")
roberta_model_path = os.path.join(current_dir, "spoiler_roberta_final") # Self-Training 后的模型

# --- 1. 加载模型 ---
print("🥊 正在加载选手...")

# 选手 1: LR (Teacher)
print("Load: LR + MPNet...")
try:
    lr_clf = joblib.load(lr_model_path)
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    if torch.cuda.is_available():
        embed_model = embed_model.to('cuda')
except Exception as e:
    print(f"LR 模型加载失败: {e}")
    exit()

# 选手 2: RoBERTa (Student)
print("Load: RoBERTa Student...")
try:
    # 显式加载，确保使用 GPU
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
    roberta_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, truncation=True, max_length=128)
except Exception as e:
    print(f"RoBERTa 模型加载失败: {e}")
    exit()

# --- 2. 加载高质量测试集 (Goodreads) ---
print("\n📚 正在加载 Goodreads 图书评论数据 (跨域测试)...")
# 我们只取其中的一部分作为测试，因为数据集太大了
# Goodreads 数据集结构通常包含 'review_sentences' 和 'has_spoiler' 标记
try:
    # 这里我们使用 streaming 模式加载，只取前 2000 个样本，防止内存爆炸
    dataset = load_dataset("wanng/goodreads-spoiler", split="test", streaming=True)
except:
    print("下载失败，尝试使用备用数据集...")
    # 备用方案：如果 wanng 的源连不上，可以用 yelp 或其他，这里假设能连上
    exit()

test_sentences = []
test_labels = []

print("正在构建测试集 (提取句子级标签)...")
# 这是一个生成器，我们取够 5000 个句子就停
counter = 0
MAX_SAMPLES = 5000

for example in dataset:
    # Goodreads 数据集的字段通常是 'review_sentences' (list of str) 和 'has_spoiler' (list of bool/int)
    # 具体字段名需要 print(example) 确认，以下是常见结构
    try:
        sents = example.get('review_sentences', [])
        labels = example.get('has_spoiler', [])
        
        if len(sents) != len(labels): continue
        
        for s, l in zip(sents, labels):
            if len(s) < 10: continue # 过滤短句
            test_sentences.append(s)
            test_labels.append(int(l)) # 0 或 1
            counter += 1
    except:
        continue
        
    if counter >= MAX_SAMPLES:
        break

print(f"✅ 测试集构建完成: {len(test_sentences)} 句")
print(f"   剧透句: {sum(test_labels)}")
print(f"   正常句: {len(test_labels) - sum(test_labels)}")

if sum(test_labels) == 0:
    print("⚠️ 警告: 测试集里没有剧透句，请检查数据集字段结构。")
    # 如果 wanng 数据集加载有问题，可以手动 mock 一些数据测试流程
    exit()

# --- 3. 比赛开始 ---

# === Round 1: LR 模型 ===
print("\n🔥 Round 1: LR (Vector) Model 推理中...")
start_time = time.time()

# 1. 计算向量
vecs = embed_model.encode(test_sentences, batch_size=128, show_progress_bar=True)
# 2. 预测
lr_preds = lr_clf.predict(vecs)

lr_time = time.time() - start_time
print(f"LR 耗时: {lr_time:.2f} 秒")

# === Round 2: RoBERTa 模型 ===
print("\n🔥 Round 2: RoBERTa (Student) Model 推理中...")
start_time = time.time()

# pipeline 自动处理 batch
roberta_results = roberta_pipe(test_sentences, batch_size=64) 
roberta_preds = [1 if res['label'] == 'LABEL_1' else 0 for res in roberta_results]

roberta_time = time.time() - start_time
print(f"RoBERTa 耗时: {roberta_time:.2f} 秒")

# --- 4. 结果对比与结算 ---

def print_metrics(name, y_true, y_pred, time_taken):
    print(f"\n📊 {name} 成绩单")
    print("-" * 30)
    print(f"推理速度 (QPS): {len(y_true)/time_taken:.2f} sentences/sec")
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")

print("\n" + "="*50)
print("🏆 最终对决结果 (Movie Models tested on Book Reviews)")
print("="*50)

print_metrics("LR Teacher (MPNet)", test_labels, lr_preds, lr_time)
print_metrics("RoBERTa Student", test_labels, roberta_preds, roberta_time)

print("="*50)

# --- 5. 错误案例分析 (谁在裸泳？) ---
print("\n🔍 差异样本分析 (Disagreement Analysis)")
count = 0
for i in range(len(test_sentences)):
    if count >= 5: break
    # 找一个 LR 对了但 RoBERTa 错了，或者反过来的例子
    if lr_preds[i] != roberta_preds[i] and test_labels[i] == 1:
        print("-" * 30)
        print(f"句子: {test_sentences[i]}")
        print(f"真实标签: {test_labels[i]}")
        print(f"LR 预测: {lr_preds[i]}")
        print(f"RoBERTa 预测: {roberta_preds[i]}")
        count += 1
