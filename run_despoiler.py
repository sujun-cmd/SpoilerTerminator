import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. 加载现代 LLM ---
# Qwen2.5-7B-Instruct: 目前开源界 7B 参数下的最强模型，指令遵循能力极强
model_id = "Qwen/Qwen2.5-7B"

print(f"🚀 正在加载 LLM: {model_id} ...")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型 (V100 显存足够跑 fp16)
# device_map="auto" 会自动把模型塞进 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --- 2. 定义智能改写函数 ---
def despoil_with_llm(spoiler_text):
    # 使用 Chat 格式的 Prompt
    messages = [
        {"role": "system", "content": "You are a professional movie editor. Your task is to rewrite movie spoilers into vague, suspenseful plot teasers for a synopsis. Never reveal who dies, who the killer is, or the specific ending. Make it sound mysterious."},
        {"role": "user", "content": f"Rewrite this spoiler into a safe teaser: '{spoiler_text}'"}
    ]
    
    # 构建 Prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        temperature=0.7,   # 稍微一点创造力
        top_p=0.9,
        do_sample=True
    )
    
    # 提取回答 (去掉 Prompt 部分)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- 3. 测试效果 ---
test_spoilers = [
    "Bruce Willis is actually a ghost in the end.",
    "Darth Vader reveals that he is Luke's father.",
    "She dies in the car accident.",
    "The killer turns out to be the detective investigating the case.",
    "They all die at the end of the movie."
]

print("\n" + "="*100)
print(f"{'Original Spoiler':<50} | {'LLM Despoiled Version (Teaser)'}")
print("="*100)

for spoiler in test_spoilers:
    safe_version = despoil_with_llm(spoiler)
    # 清理一下输出，防止模型话痨
    safe_version = safe_version.replace('"', '').strip()
    print(f"{spoiler:<50} | {safe_version}")
    print("-" * 100)



