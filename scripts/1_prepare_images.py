import os
import json

def prepare_image_dataset(image_dir):
    image_paths = []
    supported_formats = ['.jpg', '.jpeg', '.png']
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    image_dir = "./images/train2017"  # 替换为你的图片文件夹路径
    output_file = "./outputs/image_list.json"
    
    image_paths = prepare_image_dataset(image_dir)
    if not image_paths:
        print("未找到图片，请检查 image_dir 路径！")
        return
    
    os.makedirs("./outputs", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)
    
    print(f"找到 {len(image_paths)} 张图片，路径已保存至 {output_file}")

if __name__ == "__main__":
    main()