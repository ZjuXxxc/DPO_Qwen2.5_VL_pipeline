import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import gc

def load_model_7b():
    local_model_path = "models/Qwen2.5-VL-7B/snapshots/b901af65fa3b2801b73d1c5b1ff59b89d81a708f"
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor

def regenerate_sentence(model, processor, image_path, original_sentence):
    prompt = f"Generate a concise but more accurate sentence describing the image to replace the original_sentence:'{original_sentence}'."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    # 获取 prompt 的 token 数量
    prompt_length = inputs["input_ids"].shape[1]
    # 生成新内容
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=1.3,     # 更高的温度增加生成的多样性
        do_sample=True,
        top_k=10,            # 更小的 k 增加了采样的不确定性
        top_p=0.95           # 增大 top_p 提高采样的范围
    )
    # 去掉 prompt 部分，只保留新生成的 token
    new_tokens = generated_ids[0][prompt_length:]

    # 解码生成部分的 token
    output_text = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    del inputs, generated_ids
    torch.cuda.empty_cache()
    return output_text

def main():
    desc_file = "./outputs/descriptions.json"
    output_file = "./outputs/regenerated_sentences.json"
    num_variants = 4
    
    with open(desc_file, "r", encoding="utf-8") as f:
        desc_data = json.load(f)
    
    model, processor = load_model_7b()
    results = []
    
    import random
    random.seed(42)  # 设置随机种子以确保可重复性
    # 随机挑选1000个句子进行变体生成
    random.shuffle(desc_data)
    desc_data = desc_data[:100]
    # 遍历每个句子，生成变体
    for i, item in enumerate(desc_data):
        print(f"处理第 {i+1} 个句子")
        img_path = item["image"]
        sentences = item["sentences"]
        for sent in sentences:
            # 如果句子长度小于10，则跳过
            if len(sent) < 10:
                continue
            variants = []
            for _ in range(num_variants):
                new_sent = regenerate_sentence(model, processor, img_path, sent)
                variants.append(new_sent)
            results.append({
                "image": img_path,
                "original_sentence": sent,
                "variants": variants
            })
            # print(f"生成 {num_variants} 个变体：{img_path}, 原句：{sent}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"已为 {len(results)} 个句子生成变体，保存至 {output_file}")
    
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()