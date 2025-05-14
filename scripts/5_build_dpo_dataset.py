import json
import os
import logging

# 设置日志
logging.basicConfig(filename='build_dpo_dataset.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    regenerated_file = "./outputs/validated_sentences.json"  # 修正文件名
    output_file = "./outputs/dpo_data.json"
    
    # 检查输入文件
    if not os.path.exists(regenerated_file):
        logging.error(f"输入文件 {regenerated_file} 不存在")
        raise FileNotFoundError(f"未找到 {regenerated_file}")
    
    # 加载验证结果
    with open(regenerated_file, "r", encoding="utf-8") as f:
        regenerated_data = json.load(f)
    
    # 构建 DPO 数据集
    dpo_data = []
    for item in regenerated_data:
        image = item["image"]
        ranked_sentences = item.get("ranked_sentences", [])
        
        # 检查 ranked_sentences 是否有效
        if len(ranked_sentences) < 2:
            logging.warning(f"图片 {image} 的 ranked_sentences 不足 2 条，跳过")
            continue
        
        # 提取 chosen（最高分）和 rejected（最低分）
        chosen_sentence = ranked_sentences[0]["sentence"]
        rejected_sentence = ranked_sentences[-1]["sentence"]
        
        # 确保 chosen 和 rejected 不同
        if chosen_sentence == rejected_sentence:
            logging.warning(f"图片 {image} 的 chosen 和 rejected 相同，跳过")
            continue
        
        dpo_data.append({
            "image": image,
            "prompt": "Describe the image in one sentence.",
            "chosen": chosen_sentence,
            "rejected": rejected_sentence
        })
        logging.info(f"生成 DPO 对：{image}, chosen_score={ranked_sentences[0]['score']}, rejected_score={ranked_sentences[-1]['score']}")
    
    # 保存 DPO 数据集
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    print(f"DPO 数据集已保存至 {output_file}，包含 {len(dpo_data)} 条数据")
    logging.info(f"DPO 数据集已保存至 {output_file}，包含 {len(dpo_data)} 条数据")

if __name__ == "__main__":
    main()