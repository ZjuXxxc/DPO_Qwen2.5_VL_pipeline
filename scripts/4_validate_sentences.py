import json
import torch
import re
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import gc
import logging
import os

# 设置日志
logging.basicConfig(filename='validate_sentences.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_3b():
    local_model_path = "models/Qwen2.5-VL-3B/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor

def validate_sentence(model, processor, image_path, sentence):
    prompt = f"""
    You are an image-text consistency evaluator. Given a description and an image, your task is to assess how well the description matches the image. Return a single integer (0-100) indicating the **confidence** that the description is fully accurate, detailed, and not misleading, based on the visual content.

    Scoring Criteria:
    - 90-100: Description is accurate and comprehensive, with no misleading or missing key details.
    - 70-89: Generally accurate but may miss minor details or be slightly vague.
    - 40-69: Partially accurate but contains noticeable omissions or slight inaccuracies.
    - 10-39: Largely inaccurate or misleading.
    - 0-9: Completely incorrect or describes something not present at all.

    Examples:
    - Description: 'A dog runs in a park.' Image shows a dog running. Output: 95
    - Description: 'The sky is red.' Image shows a blue sky. Output: 10
    - Description: 'A cat sleeps on a couch.' Image shows a cat sleeping. Output: 90
    - Description: 'Three people sitting at a table.' Image shows two people standing. Output: 30

    Now evaluate the following description:
    '{sentence}'

    Your output (0-100, and **only the number**): 
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    # 记录 prompt 的长度
    prompt_length = inputs["input_ids"].shape[1]

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=3,do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    # 去掉 prompt 的 token，仅保留新生成部分
    new_tokens = generated_ids[0][prompt_length:]
    output_text = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    print(f"输出文本（仅新生成部分）: {output_text}")  # 调试输出

    # 正则提取数字
    match = re.search(r'\d+', output_text)
    if match:
        score = float(match.group())
        score = max(0, min(100, score))  # 限制在0-100之间
    else:
        score = 0
        logging.warning(f"无效输出: '{output_text}' for sentence: '{sentence}'")
    
    del inputs, generated_ids
    torch.cuda.empty_cache()
    return score

def main():
    regen_file = "./outputs/regenerated_sentences.json"
    output_file = "./outputs/validated_sentences.json"
    
    # 检查输入文件
    if not os.path.exists(regen_file):
        logging.error(f"输入文件 {regen_file} 不存在")
        raise FileNotFoundError(f"未找到 {regen_file}")
    
    with open(regen_file, "r", encoding="utf-8") as f:
        regen_data = json.load(f)
    
    model, processor = load_model_3b()
    results = []
    
    for item in regen_data:
        img_path = item["image"]
        original_sentence = item["original_sentence"]
        variants = item["variants"]
        
        # 验证原始句子和变体
        scores = []
        original_score = validate_sentence(model, processor, img_path, original_sentence)
        scores.append({"sentence": original_sentence, "score": original_score, "is_original": True})
        
        for variant in variants:
            score = validate_sentence(model, processor, img_path, variant)
            scores.append({"sentence": variant, "score": score, "is_original": False})
        
        # 按得分降序排序
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        
        results.append({
            "image": img_path,
            "original_sentence": original_sentence,
            "ranked_sentences": scores
        })
        logging.info(f"验证并排序：{img_path}, 原句：{original_sentence}")
        print(f"验证并排序：{img_path}, 原句：{original_sentence}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"已验证并排序 {len(results)} 个句子，保存至 {output_file}")
    logging.info(f"已验证并排序 {len(results)} 个句子，保存至 {output_file}")
    
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()