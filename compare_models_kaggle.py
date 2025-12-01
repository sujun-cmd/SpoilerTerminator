import os
import time
import json
import torch
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

# ==========================================
# 0. 环境与路径设置
# ==========================================
device = 0 if torch.cuda.is_available() else -1
current_dir = os.path.dirname(os.path.abspath(__file__))

# 模型路径
lr_model_path = os.path.join(current_dir, "vector_models", "spoiler_direction_clf.pkl")
roberta_model_path = os.path.join(current_dir, "spoiler_roberta_final")

# 数据集路径 (注意：这里使用的是解压后的 json)
DATA_FILE="/home/sujun/datasets/goodreads/2/goodreads_reviews_spoiler.json"
print(f"🚀 运行设备: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

# ==========================================
# 1. 加载两个模型 (选手入场)
# ==========================================
print("\n🥊 正在加载模型...")

# --- 选手 A: LR Teacher (基于向量) ---
print(">>> Load: LR Teacher (MPNet + Logistic Regression)...")
try:
    if not os.path.exists(lr_model_path):
        raise FileNotFoundError(f"找不到 LR 模型文件: {lr_model_path}")
    
    lr_clf = joblib.load(lr_model_path)
    
    # 加载 MPNet (用于计算句向量)
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    if torch.cuda.is_available():
        embed_model = embed_model.to('cuda')
        
except Exception as e:
    print(f"❌ LR 模型加载失败: {e}")
    exit()

# --- 选手 B: RoBERTa Student (基于微调) ---
print(">>> Load: RoBERTa Student (Fine-tuned)...")
try:
    if not os.path.exists(roberta_model_path):
        raise FileNotFoundError(f"找不到 RoBERTa 模型文件夹: {roberta_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
    
    # 使用 Pipeline 进行推理
    roberta_pipe = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        truncation=True, 
        max_length=128,
        batch_size=64
    )
except Exception as e:
    print(f"❌ RoBERTa 模型加载失败: {e}")
    exit()

# ==========================================
# 2. 加载 Kaggle Goodreads 数据
# ==========================================
print(f"\n📚 正在读取测试数据: {DATA_FILE}")

if not os.path.exists(DATA_FILE):
    print(f"❌ 错误: 找不到文件 {DATA_FILE}")
    print("请先将 'goodreads_reviews_spoiler.json' 上传到当前目录！")
    exit()

test_sentences = []
test_labels = []

# 设定测试样本数量 (太大跑得慢，5000-10000 足够出结果)
MAX_SAMPLES = 10000 
count = 0

try:
    # 使用标准 open 读取 json 文件 (JSON Lines 格式)
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 解析每一行 JSON
                review = json.loads(line)
                
                # 提取 review_sentences 字段
                # 格式: [[label, text], [label, text], ...]
                # label: 0 (safe), 1 (spoiler)
                sentences_data = review.get('review_sentences', [])
                
                for label, text in sentences_data:
                    # 过滤掉太短的句子 (噪音)
                    if len(text) < 10: continue
                    
                    test_sentences.append(text)
                    test_labels.append(int(label))
                    count += 1
                    
            except Exception as e:
                continue
            
            # 达到数量限制则停止
            if count >= MAX_SAMPLES:
                break
                
except Exception as e:
    print(f"读取数据时出错: {e}")
    exit()

print(f"✅ 测试集构建完成: {len(test_sentences)} 句")
print(f"   剧透句 (Label 1): {sum(test_labels)}")
print(f"   正常句 (Label 0): {len(test_labels) - sum(test_labels)}")
spoiler_ratio = sum(test_labels) / len(test_labels)
print(f"   剧透占比: {spoiler_ratio:.2%}")

if sum(test_labels) == 0:
    print("⚠️ 警告: 测试集里没有剧透句，评估结果可能无效。")

# ==========================================
# 3. 比赛开始 (Inference)
# ==========================================

# --- Round 1: LR 模型 ---
print("\n🔥 Round 1: LR (Vector) Model 推理中...")
start_time = time.time()

# 1. 计算向量 (Batch size 128)
vecs = embed_model.encode(test_sentences, batch_size=128, show_progress_bar=True)
# 2. 预测
lr_preds = lr_clf.predict(vecs)

lr_time = time.time() - start_time
print(f"LR 耗时: {lr_time:.2f} 秒")

# --- Round 2: RoBERTa 模型 ---
print("\n🔥 Round 2: RoBERTa (Student) Model 推理中...")
start_time = time.time()

# Pipeline 推理
roberta_results = roberta_pipe(test_sentences)
# 提取标签 (LABEL_1 -> 1, LABEL_0 -> 0)
roberta_preds = [1 if res['label'] == 'LABEL_1' else 0 for res in roberta_results]

roberta_time = time.time() - start_time
print(f"RoBERTa 耗时: {roberta_time:.2f} 秒")

# ==========================================
# 4. 结果对比与结算
# ==========================================

def print_metrics(name, y_true, y_pred, time_taken):
    print(f"\n📊 {name} 成绩单")
    print("-" * 40)
    print(f"推理速度 (QPS): {len(y_true)/time_taken:.2f} sentences/sec")
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")

print("\n" + "="*60)
print("🏆 最终对决结果 (Movie Models -> Book Reviews)")
print("="*60)

print_metrics("LR Teacher (MPNet Vectors)", test_labels, lr_preds, lr_time)
print_metrics("RoBERTa Student (Fine-tuned)", test_labels, roberta_preds, roberta_time)

print("="*60)

# ==========================================
# 5. 差异分析 (Disagreement Analysis)
# ==========================================
print("\n🔍 差异样本分析 (看谁更准？)")
print("只显示：真实标签是剧透，但模型产生了分歧的例子")
print("-" * 60)

count = 0
for i in range(len(test_sentences)):
    if count >= 10: break
    
    # 我们只关心真实的剧透句子 (Label 1)
    if test_labels[i] == 1:
        # 如果两模型预测结果不同
        if lr_preds[i] != roberta_preds[i]:
            lr_res = "✅" if lr_preds[i]==1 else "❌"
            rob_res = "✅" if roberta_preds[i]==1 else "❌"
            
            print(f"句子: {test_sentences[i][:150]}...") # 截断显示
            print(f"LR 预测: {lr_res} ({lr_preds[i]}) | RoBERTa 预测: {rob_res} ({roberta_preds[i]})")
            print("-" * 30)
            count += 1
