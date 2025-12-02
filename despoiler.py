import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class SpoilerRewriter:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct"):
        """
        初始化改写器，加载 4-bit 量化的 LLM。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [Rewriter] 正在加载改写模型: {model_id} ...")
        
        try:
            # V100 专用配置: 4-bit 量化 + Float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print("✅ [Rewriter] 模型加载成功！")
            
        except Exception as e:
            print(f"❌ [Rewriter] 模型加载失败: {e}")
            raise e

    def rewrite(self, text):
        """
        输入一段剧透文本，返回改写后的 Teaser。
        """
        messages = [
            {"role": "system", "content": "You are a professional movie editor. Rewrite the spoiler into a vague, suspenseful plot teaser. You must retain the initial emotional bias (positive/negative review). Do NOT reveal names of characters who die, the killer's identity, or the specific ending. Keep it concise."},
            {"role": "user", "content": f"Rewrite this spoiler: '{text}'"}
        ]
        
        # 构建 Chat 格式 Prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        # 获取 input 长度以便切片，只返回生成的答案
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=64, # 改写不需要太长
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 切片：去掉 Prompt，只留回答
        output_ids = generated_ids[0][input_len:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return response.strip()

# 如果直接运行此文件，进行简单测试
if __name__ == "__main__":
    rewriter = SpoilerRewriter()
    test_sent = "Bruce Willis is actually a ghost at the end."
    print(f"Original: {test_sent}")
    print(f"Rewritten: {rewriter.rewrite(test_sent)}")