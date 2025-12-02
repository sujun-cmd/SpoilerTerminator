import nltk
import os
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, DatasetDict
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)

# --- 1. 环境准备 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- 2. 模型配置 ---
# 切换为 BERT-Mini (prajjwal1/bert-mini)
# 参数量仅约 11M (RoBERTa-base 是 125M)，速度极快
model_name = "prajjwal1/bert-mini" 
hf_model = model_name

print(f"🚀 正在启动极速模型: {model_name}")

# --- 3. 关键词列表 (保持不变) ---
spoiler_keywords = [
    "plot twist", "twist reveal", "big reveal", "the truth is", "it turns out", "actually",
    "in reality", "hidden agenda", "secret identity", "real identity", "double identity",
    "backstory revealed", "the real reason", "foreshadowing pays off", "major turning point",
    "unexpected shift", "game changer", "crucial clue", "final clue", "the moment everything changes",
    "true intention", "the real plan", "flashback explains", "surprise appearance", "cameo reveal",
    "dies", "doesn't make it", "survives", "comes back to life", "revived",
    "was alive the whole time", "not dead", "betrays", "betrayal", "turns evil",
    "goes dark", "redemption arc", "secret sibling", "long-lost sibling", "hidden family",
    "adoption reveal", "fake death", "sacrifice", "identity swap", "body double",
    "double agent", "undercover reveal", "ending explained", "final scene", "post-credits scene",
    "after-credits reveal", "true ending", "hidden ending", "alternate ending", "final twist",
    "cliffhanger ending", "resolution", "full explanation", "gets together with", "ends up with",
    "confession scene", "love triangle resolved", "breakup", "proposal", "wedding reveal",
    "secret crush reveal", "the killer is", "the murderer is", "the culprit", "the mastermind",
    "inside job", "was planned all along", "false alibi", "unreliable narrator", "hallucination",
    "imaginary character", "not real", "dream sequence", "it was all a dream", "experiment failure",
    "switcheroo", "identity twist", "secret recording", "prophecy fulfilled", "chosen one reveal",
    "time loop revealed", "parallel universe twist", "alternate timeline", "memory wipe", "mind control",
    "the artifact's power", "true nature of the world", "simulation reveal", "the letter says",
    "the box contains", "the map shows", "the key unlocks", "hidden message", "coded message",
    "ancient secret", "the experiment worked", "the experiment failed", "major spoilers",
    "spoiler alert", "big spoiler ahead", "full plot summary", "ending breakdown",
    "all secrets explained", "here’s what really happened", "you won’t believe this part"
]

# --- 4. 数据加载与处理 ---
print("正在加载数据集...")
raw_dataset = load_dataset("bhavyagiri/imdb-spoiler")

# 划分数据集
split_1 = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
test_dataset = split_1["test"]
split_2 = split_1["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_2["train"]
eval_dataset = split_2["test"]

dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
    "test": test_dataset
})

def process_data(batch):
    new_sentences = []
    new_labels = []
    for text, doc_label in zip(batch["text"], batch["label"]):
        sents = sent_tokenize(text)
        total_sents = len(sents)
        for sent in sents:
            if len(sent) < 5: continue
            label = 0
            if doc_label == 1:
                if total_sents == 1:
                    label = 1
                else:
                    # 关键词匹配逻辑
                    sent_lower = sent.lower()
                    for kw in spoiler_keywords:
                        if kw in sent_lower:
                            label = 1
                            break
            new_sentences.append(sent)
            new_labels.append(label)
    return {"sentence": new_sentences, "label": new_labels}

print("正在处理数据 (分句与打标)...")
processed_dataset = dataset.map(
    process_data, 
    batched=True, 
    batch_size=1000, 
    remove_columns=dataset["train"].column_names
)

# --- 5. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(hf_model)

def tokenize_function(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

print("Tokenizing...")
tokenized = processed_dataset.map(tokenize_function, batched=True)

# --- 6. 定义指标计算函数 ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # 计算 Precision, Recall, F1 (binary 模式用于二分类)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 7. 训练设置 ---
model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=2)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "results_mini") # 修改目录名
model_dir = os.path.join(current_dir, "spoiler_model_mini")

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    
    # 极速优化：BERT-Mini 很小，Batch Size 可以开得很大
    per_device_train_batch_size=128,    
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
    
    num_train_epochs=3,
    learning_rate=5e-5, # 小模型可以承受稍高的学习率
    weight_decay=0.01,
    fp16=False, 
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --- 8. 执行训练与评估 ---
print("开始极速训练 BERT-Mini...")
trainer.train()

# 保存最终模型
trainer.save_model(model_dir)

# --- 9. 最终测试并保存到 eval.txt ---
print("正在进行最终测试集评估...")
test_metrics = trainer.evaluate(tokenized["test"])

eval_file_path = os.path.join(current_dir, "eval.txt")

print(f"将结果写入: {eval_file_path}")
with open(eval_file_path, "w", encoding="utf-8") as f:
    f.write(f"Model: {model_name}\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy:  {test_metrics['eval_accuracy']:.4f}\n")
    f.write(f"Precision: {test_metrics['eval_precision']:.4f}\n")
    f.write(f"Recall:    {test_metrics['eval_recall']:.4f}\n")
    f.write(f"F1 Score:  {test_metrics['eval_f1']:.4f}\n")
    f.write("-" * 30 + "\n")
    f.write(f"Loss:      {test_metrics['eval_loss']:.4f}\n")

print("搞定！结果已保存。")