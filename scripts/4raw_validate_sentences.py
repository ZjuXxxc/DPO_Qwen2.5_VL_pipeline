import json
import torch
from transformers import AutoProcessor , Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import nltk

# nltk.download('punkt')

def load_model_7b():
    local_model_path = "models/Qwen2.5-VL-7B/snapshots/b901af65fa3b2801b73d1c5b1ff59b89d81a708f"
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model,processor

def regenerate_sentence(model, processor, image_path, rejected_sentence):
    prompt = f"The following sentence does not accurately describe the image content:'{rejected_sentence}'.Please provide a more accurate description sentence of similar length that may be different:"
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
    return output_text[0]

def main():
    validity_file = "./outputs/sentence_validity.json"
    output_file = "./outputs/regenerated_sentences.json"
    
    # 加载有效性结果
    with open(validity_file, "r", encoding="utf-8") as f:
        validity_data = json.load(f)
    
    # 加载 7B 模型
    model, processor = load_model_7b()
    
    # 处理不符合的句子
    results = []
    for item in validity_data:
        if not item["is_valid"]:
            img_path = item["image"]
            rejected = item["sentence"]
            regenerated = regenerate_sentence(model, processor, img_path, rejected)
            results.append({
                "image": img_path,
                "rejected": rejected,
                "chosen": regenerated
            })
            print(f"重生成：{rejected} -> {regenerated}")
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"重生成句子已保存至 {output_file}")

if __name__ == "__main__":
    main()