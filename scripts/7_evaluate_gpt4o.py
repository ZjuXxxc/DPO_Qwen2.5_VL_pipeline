import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import base64
import os
import gc

def load_model_7b(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor

def generate_description(model, processor, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe the image in one sentence."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    del inputs, generated_ids
    torch.cuda.empty_cache()
    return output_text

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_with_gpt4o(image_path, description, api_key):
    client = OpenAI(api_key=api_key)
    
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Is the following description accurate for the image? '{description}' Rate its accuracy from 0 to 100."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=10
    )
    
    try:
        score = float(response.choices[0].message.content)
        score = max(0, min(100, score))
    except ValueError:
        score = 0
    
    return score

def main():
    image_list_file = "./outputs/image_list.json"
    pretrain_model_path = "models/Qwen2.5-VL-7B/snapshots/b901af65fa3b2801b73d1c5b1ff59b89d81a708f"
    dpo_model_path = "./models/Qwen2.5-VL-7B-DPO/"
    output_file = "./outputs/evaluation_results.json"
    api_key = os.getenv("OPENAI_API_KEY")  # 设置你的 API 密钥
    
    with open(image_list_file, "r", encoding="utf-8") as f:
        image_paths = json.load(f)[:10]  # 测试 10 张图片
    
    pretrain_model, pretrain_processor = load_model_7b(pretrain_model_path)
    dpo_model, dpo_processor = load_model_7b(dpo_model_path)
    
    results = []
    
    for img_path in image_paths:
        pretrain_desc = generate_description(pretrain_model, pretrain_processor, img_path)
        dpo_desc = generate_description(dpo_model, dpo_processor, img_path)
        
        pretrain_score = evaluate_with_gpt4o(img_path, pretrain_desc, api_key)
        dpo_score = evaluate_with_gpt4o(img_path, dpo_desc, api_key)
        
        results.append({
            "image": img_path,
            "pretrain_description": pretrain_desc,
            "pretrain_score": pretrain_score,
            "dpo_description": dpo_desc,
            "dpo_score": dpo_score
        })
        print(f"评估：{img_path}, Pretrain: {pretrain_score}, DPO: {dpo_score}")
    
    avg_pretrain = sum(r["pretrain_score"] for r in results) / len(results)
    avg_dpo = sum(r["dpo_score"] for r in results) / len(results)
    
    output_data = {
        "average_pretrain_score": avg_pretrain,
        "average_dpo_score": avg_dpo,
        "results": results
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，平均得分：Pretrain={avg_pretrain:.2f}, DPO={avg_dpo:.2f}")
    print(f"结果保存至 {output_file}")
    
    del pretrain_model, pretrain_processor, dpo_model, dpo_processor
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()