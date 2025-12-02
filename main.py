import os
import json
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    pipeline, 
    BitsAndBytesConfig
)
from tqdm import tqdm

# ==========================================
# 0. 配置与环境
# ==========================================
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 运行设备: {'GPU' if device==0 else 'CPU'}")

# 路径设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(CURRENT_DIR, "spoiler_roberta_final") # 你的检测模型
REWRITER_ID = "Qwen/Qwen2.5-7B-Instruct"                           # 你的改写模型

INPUT_FILE = os.path.join(CURRENT_DIR, "reviews.txt")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "despoiled_reviews.txt")

# 确保 NLTK 数据存在
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================================
# 1. 加载模型 (Detector & Rewriter)
# ==========================================

print("\n📦 正在加载模型...")

# --- A. 加载剧透检测器 (RoBERTa) ---
print(f"   [1/2] Loading Detector: {DETECTOR_PATH} ...")
try:
    det_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
    det_model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH)
    # 使用 pipeline 加速推理
    detector = pipeline(
        "text-classification", 
        model=det_model, 
        tokenizer=det_tokenizer, 
        device=device, 
        truncation=True, 
        max_length=128
    )
except Exception as e:
    print(f"❌ 检测模型加载失败: {e}")
    exit()

# --- B. 加载剧透改写器 (Qwen LLM) ---
print(f"   [2/2] Loading Rewriter: {REWRITER_ID} ...")
try:
    # 4-bit 量化配置 (为了省显存)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # V100 关键设置！不要用 bfloat16
)
    
    rw_tokenizer = AutoTokenizer.from_pretrained(REWRITER_ID)
    rw_model = AutoModelForCausalLM.from_pretrained(
        REWRITER_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
except Exception as e:
    print(f"❌ 改写模型加载失败: {e}")
    exit()

# ==========================================
# 2. 定义功能函数
# ==========================================

def rewrite_spoiler(text):
    """使用 LLM 改写剧透句子"""
    messages = [
        {"role": "system", "content": "You are a professional movie editor. Rewrite the spoiler into a vague, suspenseful plot teaser. Do NOT reveal names of characters who die, the killer's identity, or the specific ending. Keep it concise."},
        {"role": "user", "content": f"Rewrite this spoiler: '{text}'"}
    ]
    
    prompt = rw_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = rw_tokenizer([prompt], return_tensors="pt").to(rw_model.device)
    
    # 获取 input 长度以便切片
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = rw_model.generate(
            **inputs,
            max_new_tokens=64, # 改写不需要太长
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # 只提取生成的回复部分
    output_ids = generated_ids[0][input_len:]
    response = rw_tokenizer.decode(output_ids, skip_special_tokens=True)
    return response.strip()

def process_single_review(review_json):
    """处理单条评论：分句 -> 检测 -> 改写 -> 重组"""
    original_text = review_json.get("text", "")
    if not original_text:
        return None

    sentences = sent_tokenize(original_text)
    processed_sentences = []
    
    # 标记这篇评论是否包含剧透
    has_spoiler = False
    
    # 批量检测 (虽然这里是逐句循环，但对于长评论可以先攒 batch，这里为了逻辑清晰逐句处理)
    # 对于生产环境，建议先 flatten 所有句子做 batch inference
    
    # 获取检测结果
    preds = detector(sentences)
    
    for sent, pred in zip(sentences, preds):
        label = pred['label'] # LABEL_0 (Safe) or LABEL_1 (Spoiler)
        score = pred['score']
        
        # 设定阈值：如果模型非常有信心是剧透 (>0.8)，或者是 LABEL_1
        is_spoiler_sent = (label == 'LABEL_1')
        
        if is_spoiler_sent:
            has_spoiler = True
            print(f"   🚨 发现剧透: {sent[:50]}...")
            # 调用 LLM 改写
            safe_version = rewrite_spoiler(sent)
            print(f"      ✨ 改写为: {safe_version[:50]}...")
            processed_sentences.append(safe_version)
        else:
            processed_sentences.append(sent)
            
    # 重组评论
    final_text = " ".join(processed_sentences)
    
    return {
        "original_text": original_text,
        "processed_text": final_text,
        "is_spoiler_review": has_spoiler,
        "original_json": review_json # 保留原始元数据
    }

# ==========================================
# 3. 主流程
# ==========================================

def main():
    print(f"\n🚀 开始处理文件: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ 找不到输入文件！请先创建 reviews.txt")
        return

    results = []
    
    # 读取所有行
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"📚 总共有 {len(lines)} 条评论待处理...")
    
    # 进度条处理
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
                
            # 处理
            result = process_single_review(data)
            
            if result:
                # 写入结果文件 (JSON Lines 格式)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                # 强制刷新缓冲区，防止程序中断丢失数据
                f_out.flush()
                
    print(f"\n✅ 处理完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()