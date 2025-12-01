import nltk
import os
import torch
import torch.nn as nn
import numpy as np
import evaluate
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    pipeline
)

# --- 0. 环境设置 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir_path = os.path.join(current_dir, "results_hybrid")
model_save_path = os.path.join(current_dir, "spoiler_roberta_hybrid")

# --- 1. 关键词 (保底策略) ---
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
    "all secrets explained", "here’s what really happened", "you won’t believe this part","dead","died","defeat"
]

# --- 2. AI 老师 (补漏策略) ---
# 使用 pipeline 简化推理，防止 index 搞错
print("🚀 正在加载 AI 辅助模型...")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=device)

def ai_check_batch(sentences):
    """
    使用 Zero-Shot 分类来判断是否是剧透
    """
    if not sentences:
        return []
    
    candidate_labels = ["spoiler", "safe"]
    hypothesis_template = "This sentence contains a {}."
    
    # 批量推理
    results = classifier(sentences, candidate_labels, hypothesis_template=hypothesis_template)
    
    # 解析结果: 如果 label[0] 是 spoiler，则为 1
    labels = [1 if res['labels'][0] == 'spoiler' else 0 for res in results]
    return labels

# --- 3. 加载数据 ---
raw_dataset = load_dataset("bhavyagiri/imdb-spoiler")
# 仅供测试，如果想跑全量请注释下面这行
# raw_dataset["train"] = raw_dataset["train"].select(range(5000))

split_1 = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_1["train"],
    "validation": split_1["test"]
})

# --- 4. 混合数据清洗逻辑 ---
def process_data_hybrid(batch):
    final_sentences = []
    final_labels = []
    
    # 暂存需要 AI 判断的句子
    ai_candidates = []
    ai_indices = [] # 记录它们在 final_sentences 里的位置，以便回填
    
    for text, doc_label in zip(batch["text"], batch["label"]):
        sents = sent_tokenize(text)
        
        for sent in sents:
            if len(sent) < 5: continue
            
            # 1. 正常评论 -> 全是 0
            if doc_label == 0:
                final_sentences.append(sent)
                final_labels.append(0)
                continue
            
            # 2. 剧透评论 -> 混合检查
            sent_lower = sent.lower()
            
            # 策略 A: 关键词命中 -> 肯定是 1 (High Precision)
            keyword_hit = False
            for kw in spoiler_keywords:
                if kw in sent_lower:
                    keyword_hit = True
                    break
            
            if keyword_hit:
                final_sentences.append(sent)
                final_labels.append(1)
            else:
                # 策略 B: 没命中关键词 -> 放入待定区，等 AI 判
                # 先占个位，填 -1
                final_sentences.append(sent)
                final_labels.append(-1)
                ai_candidates.append(sent)
                ai_indices.append(len(final_labels) - 1)
    
    # 3. 批量 AI 判决
    if ai_candidates:
        # 这里为了防止 OOM，可以再分个小 batch，或者直接交给 pipeline 处理
        # pipeline 默认处理列表很稳
        ai_results = ai_check_batch(ai_candidates)
        
        # 回填结果
        for idx, label in zip(ai_indices, ai_results):
            final_labels[idx] = label
            
    # 4. 过滤掉 AI 认为是 0 的句子 (在剧透评论里，如果 AI 和关键词都说是 0，那就丢弃，防止噪音)
    #    但为了防止 F1=0，我们暂时保留它们作为 0，或者丢弃
    #    这里选择：丢弃 (Label 0)，只保留 正常评论里的0 和 剧透评论里的1
    
    filtered_sentences = []
    filtered_labels = []
    
    for s, l in zip(final_sentences, final_labels):
        if l == 1:
            filtered_sentences.append(s)
            filtered_labels.append(1)
        elif l == 0:
            # 这里的 0 大部分来自 doc_label=0，少量来自 AI 判为 safe
            filtered_sentences.append(s)
            filtered_labels.append(0)
        # l == -1 的情况已经被填了，如果 AI 判为 0 (safe) 且来源是 doc_label=1
        # 这种句子是 "剧透评论里的废话"，我们丢弃它！
            
    return {"sentence": filtered_sentences, "label": filtered_labels}

print("正在执行混合策略清洗 (Keywords + AI)...")
processed_dataset = dataset.map(
    process_data_hybrid, 
    batched=True, 
    batch_size=50, # 调小一点，因为里面有模型推理
    remove_columns=dataset["train"].column_names
)

# === 关键检查点 ===
train_labels = processed_dataset["train"]["label"]
pos_count = sum(train_labels)
neg_count = len(train_labels) - pos_count

print("="*40)
print(f"最终数据集统计:")
print(f"总句数: {len(train_labels)}")
print(f"剧透句 (Label 1): {pos_count}")
print(f"正常句 (Label 0): {neg_count}")
print("="*40)

if pos_count < 100:
    print("🚨 严重警告: 正样本太少！模型根本学不到东西。")
    print("可能原因: 关键词没匹配上，且 AI 模型也没识别出来。")
    # 强制设置一个权重，虽然可能没用
    ratio = 100.0
else:
    ratio = neg_count / pos_count
    print(f"计算出的正样本权重: {ratio:.2f}")

class_weights = torch.tensor([1.0, ratio]).float()

# --- 5. 正式训练 ---
model_name = "roberta-large" # 用 Base 吧，V100 跑 Large 有点慢，先跑通 Base 拿分
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

tokenized = processed_dataset.map(preprocess, batched=True)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = class_weights.to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 加载多个指标
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

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir=output_dir_path,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True, # V100 开启
    bf16=False,
    push_to_hub=False
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("开始训练...")
trainer.train()

print("评估中...")
metrics = trainer.evaluate()
print(f"Final F1: {metrics['eval_f1']:.4f}")
print(f"Final Recall: {metrics['eval_recall']:.4f}")

trainer.save_model(model_save_path)
