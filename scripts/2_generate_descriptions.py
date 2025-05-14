import json
import torch
import random
from transformers import AutoProcessor , Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import gc
import re
import nltk

def load_model_7b():
    local_model_path = "models/Qwen2.5-VL-7B/snapshots/b901af65fa3b2801b73d1c5b1ff59b89d81a708f"
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model,processor

def generate_description(model, processor , image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": "Describe the image in detail, including objects, attributes, and positional relationships.",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = model.generate(**inputs,max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    description = output_text[0].strip() if output_text else ""
    if description and not description.endswith('.'):
        sentences = re.split(r'(?<=[.!?])\s+', description)
        if len(sentences) > 1:
            description = ' '.join(sentences[:-1]).strip()
        else:
            description = ""
        print(f"警告：描述不完整（{image_path}），已移除最后一句话：{description}")
    return description

def main():
    # 配置参数
    image_list_file = "./outputs/image_list.json"
    config_file = "./outputs/config.json"
    descriptions_file = "./outputs/descriptions.json"
    num_images = 1000  # 指定使用的图片数量 n，可自行修改
    random_seed = 42  # 固定随机种子，可自行修改

    # 设置随机种子，确保可重复性
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 加载图片路径
    with open(image_list_file, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
    
    # 随机选择 n 张图片
    if num_images > len(image_paths):
        print(f"警告：请求的图片数量 {num_images} 超过可用图片 {len(image_paths)}，将使用全部图片")
        num_images = len(image_paths)
    selected_paths = random.sample(image_paths, num_images)
    
    # 加载模型
    model, processor = load_model_7b()
    
    # 生成描述
    results = []
    for img_path in selected_paths:
        desc = generate_description(model, processor, img_path)
        sentences = nltk.sent_tokenize(desc)
        results.append({"image": img_path, "description": desc, "sentences": sentences})
        print(f"生成描述：{img_path}")

    # 保存参数 包括生成模型 随机种子参数 图片数量
    config = {
        "num_images": num_images,
        "random_seed": random_seed,
        "model": "Qwen2.5-VL-7B",
        "output_file": descriptions_file
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)    
    # 保存结果
    with open(descriptions_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"已为 {len(results)} 张图片生成描述，保存至 {descriptions_file}")
    # 清理模型
    del model,processor,results
    torch.cuda.empty_cache()
    gc.collect()
    print("模型已清理，缓存已释放")

if __name__ == "__main__":
    main()