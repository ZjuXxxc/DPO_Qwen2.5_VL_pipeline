import json
import torch
import nltk
from transformers import AutoProcessor , Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

# nltk.download('punkt_tab')

def load_model_3b():
    local_model_path = "models/Qwen2.5-VL-3B/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor

def split_into_sentences(description):
    sentences = nltk.sent_tokenize(description)
    return [s.strip() for s in sentences if s.strip()]

def check_sentence_validity(model, processor, image_path, sentence):
    prompt = f"Please decide whether the following sentence accurately describes the content of the picture: '{sentence}'. Answer 'Yes.' or 'No.'."
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
                    "text": prompt,
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
    print(f"模型输出：{output_text}")
    return output_text[0] == "Yes."

def main():
    descriptions_file = "./outputs/descriptions.json"
    output_file = "./outputs/sentence_validity.json"
    
    # 加载描述
    with open(descriptions_file, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    # 加载 3B 模型
    model, processor = load_model_3b()
    
    # 处理每个描述
    results = []
    for item in descriptions:
        img_path = item["image"]
        desc = item["description"]
        sentences = split_into_sentences(desc)
        
        for sent in sentences:
            is_valid = check_sentence_validity(model, processor, img_path, sent)
            results.append({
                "image": img_path,
                "sentence": sent,
                "is_valid": is_valid
            })
            print(f"检查句子：{sent} -> {'符合' if is_valid else '不符合'}")
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"句子有效性结果已保存至 {output_file}")

if __name__ == "__main__":
    main()