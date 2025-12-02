import os
import json
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# 引入我们刚才写的模块
from despoiler import SpoilerRewriter

# ==========================================
# 0. 配置与环境
# ==========================================
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 主程序运行设备: {'GPU' if device==0 else 'CPU'}")

# 路径设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 你的检测模型路径 (RoBERTa Student)
DETECTOR_PATH = os.path.join(CURRENT_DIR, "spoiler_roberta_final") 

INPUT_FILE = os.path.join(CURRENT_DIR, "reviews.txt")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "despoiled_reviews.txt")

# 确保 NLTK 数据存在
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================================
# 1. 加载模型
# ==========================================

print("\n📦 初始化 Pipeline...")

# --- A. 加载剧透检测器 (RoBERTa) ---
print(f"   [1/2] Loading Detector: {DETECTOR_PATH} ...")
try:
    if not os.path.exists(DETECTOR_PATH):
        raise FileNotFoundError(f"找不到检测模型: {DETECTOR_PATH}")

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
    print("✅ [Detector] 加载成功")
except Exception as e:
    print(f"❌ [Detector] 加载失败: {e}")
    exit()

# --- B. 加载剧透改写器 (从 despoiler.py) ---
print(f"   [2/2] Loading Rewriter (LLM) ...")
try:
    # 实例化我们在 despoiler.py 里写的类
    # 默认加载 Qwen2.5-14B-Instruct
    rewriter = SpoilerRewriter() 
except Exception as e:
    print(f"❌ [Rewriter] 初始化失败，请检查显存或 despoiler.py: {e}")
    exit()

# ==========================================
# 2. 核心处理逻辑
# ==========================================

def process_single_review(review_json):
    """处理单条评论：分句 -> 检测 -> 改写 -> 重组"""
    original_text = review_json.get("text", "")
    if not original_text:
        return None

    sentences = sent_tokenize(original_text)
    processed_sentences = []
    
    # 标记这篇评论是否包含剧透
    has_spoiler = False
    
    # 批量检测 (传入 list)
    try:
        preds = detector(sentences)
    except Exception as e:
        print(f"⚠️ 检测出错: {e}, 跳过此评论")
        return None
    
    for sent, pred in zip(sentences, preds):
        label = pred['label'] # LABEL_0 (Safe) or LABEL_1 (Spoiler)
        # score = pred['score']
        
        # 判断逻辑: LABEL_1 为剧透
        is_spoiler_sent = (label == 'LABEL_1')
        
        if is_spoiler_sent:
            has_spoiler = True
            # 为了日志好看，只打印前50个字符
            clean_sent = sent.replace('\n', ' ')
            print(f"   🚨 发现剧透: {clean_sent[:50]}...")
            
            # 调用 LLM 改写
            safe_version = rewriter.rewrite(sent)
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
        "original_json": review_json
    }

# ==========================================
# 3. 主循环
# ==========================================

def main():
    print(f"\n🚀 开始处理文件: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ 找不到输入文件！请先确保 reviews.txt 存在。")
        return

    # 读取所有行
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"📚 待处理评论数: {len(lines)}")
    
    # 进度条处理
    processed_count = 0
    spoiler_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines, desc="Processing"):
            try:
                line = line.strip()
                if not line: continue
                
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            # 处理
            result = process_single_review(data)
            
            if result:
                # 写入结果文件
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush() # 实时保存
                
                processed_count += 1
                if result['is_spoiler_review']:
                    spoiler_count += 1
                
    print(f"\n✅ 全部完成！")
    print(f"   - 总处理: {processed_count}")
    print(f"   - 含剧透: {spoiler_count}")
    print(f"   - 结果保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()