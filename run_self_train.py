import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)

# --- 0. 环境与配置 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

current_dir = os.path.dirname(os.path.abspath(__file__))
# 之前保存 LR 模型的目录
vector_model_dir = os.path.join(current_dir, "vector_models")
# 最终模型保存目录
final_output_dir = os.path.join(current_dir, "results_final_roberta")
final_save_path = os.path.join(current_dir, "spoiler_roberta_final")

# --- 1. 召唤“老师” (加载 LR 和 句向量模型) ---
print("🎓 正在加载 Teacher 模型 (LR + SentenceBERT)...")
clf_path = os.path.join(vector_model_dir, "spoiler_direction_clf.pkl")

if not os.path.exists(clf_path):
    raise FileNotFoundError("找不到 LR 模型！请先运行之前的 vector 训练脚本。")

teacher_clf = joblib.load(clf_path)

embed_model = SentenceTransformer('all-mpnet-base-v2')
if torch.cuda.is_available():
    embed_model = embed_model.to('cuda')

# --- 2. 准备原始数据 ---
print("📦 正在加载原始数据集...")
raw_dataset = load_dataset("bhavyagiri/imdb-spoiler")

# 切分出验证集 (这部分不动，用来考试)
split = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
raw_train = split["train"]
raw_val = split["test"] # 验证集

# --- 3. 老师清洗数据 (The Purge) ---
print("�� 老师正在批改作业 (数据清洗中)...")

# 3.1 提取所有句子
all_sentences = []
# 为了追踪进度，我们先不分 batch，直接把所有句子拿出来
# 这是一个内存密集型操作，但 V100 节点内存通常够用
print("正在切分句子...")
for text in raw_train["text"]:
    sents = sent_tokenize(text)
    for sent in sents:
        if len(sent) < 8: continue # 太短的不要
        all_sentences.append(sent)

print(f"原始句子总数: {len(all_sentences)}")

# 3.2 计算向量 (Batch 处理)
print("正在计算向量 (这可能需要几分钟)...")
# encode 自动处理 batching
embeddings = embed_model.encode(all_sentences, batch_size=128, show_progress_bar=True)

# 3.3 老师打分
print("老师正在打分...")
probs = teacher_clf.predict_proba(embeddings)[:, 1] # 获取“剧透概率”

# 3.4 严格筛选 (阈值过滤)
# 策略：只保留极其确定的样本
HIGH_CONFIDENCE_SPOILER = 0.7  # 高于这个才是剧透
HIGH_CONFIDENCE_SAFE = 0.40     # 低于这个才是安全
# 中间的 (0.4 ~ 0.7) 全部丢弃！

clean_sentences = []
clean_labels = []

print(f"正在应用阈值过滤 (Safe < {HIGH_CONFIDENCE_SAFE}, Spoiler > {HIGH_CONFIDENCE_SPOILER})...")

for sent, score in zip(all_sentences, probs):
    if score > HIGH_CONFIDENCE_SPOILER:
        clean_sentences.append(sent)
        clean_labels.append(1)
    elif score < HIGH_CONFIDENCE_SAFE:
        clean_sentences.append(sent)
        clean_labels.append(0)
    # else: 丢弃模棱两可的

# --- 4. 构建纯净数据集 ---
clean_pos = sum(clean_labels)
clean_neg = len(clean_labels) - clean_pos

print("="*40)
print("✨ 清洗完成！数据统计 ✨")
print(f"保留总数: {len(clean_labels)} (丢弃了 {len(all_sentences) - len(clean_labels)} 句废话)")
print(f"纯净剧透句 (Label 1): {clean_pos}")
print(f"纯净安全句 (Label 0): {clean_neg}")
ratio = clean_neg / clean_pos if clean_pos > 0 else 1.0
print(f"正负比例: 1 : {ratio:.2f}")
print("="*40)

# 转回 HuggingFace Dataset
train_dataset = Dataset.from_dict({"sentence": clean_sentences, "label": clean_labels})

# 处理验证集 (验证集我们不做清洗，保持原样，或者简单切分，为了公平评估)
# 这里为了代码简单，我们对验证集只做简单切分，使用原始 Weak Label 作为参考
# (虽然验证集也有噪音，但它是我们唯一的参考标准)
val_sentences = []
val_labels = []
for text, label in zip(raw_val["text"], raw_val["label"]):
    for sent in sent_tokenize(text):
        if len(sent) < 8: continue
        val_sentences.append(sent)
        val_labels.append(label)
val_dataset = Dataset.from_dict({"sentence": val_sentences, "label": val_labels})

# --- 5. 训练学生 (RoBERTa) ---
print("👨‍🎓 学生 (RoBERTa) 开始学习...")
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_val = val_dataset.map(preprocess, batched=True)

# 权重
class_weights = torch.tensor([1.0, ratio]).float()

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = class_weights.to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 指标
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    p = precision_metric.compute(predictions=preds, references=labels)["precision"]
    r = recall_metric.compute(predictions=preds, references=labels)["recall"]
    return {"f1": f1, "precision": p, "recall": r}

training_args = TrainingArguments(
    output_dir=final_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    per_device_train_batch_size=32, # V100 性能全开
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True, # V100 必备
    bf16=False,
    push_to_hub=False
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

print("📝 最终考试 (评估)...")
metrics = trainer.evaluate()
print(f"Final F1: {metrics['eval_f1']:.4f}")
print(f"Final Precision: {metrics['eval_precision']:.4f}")
print(f"Final Recall: {metrics['eval_recall']:.4f}")

trainer.save_model(final_save_path)
print("🎉 毕业了！Self-Training 流程结束。")
