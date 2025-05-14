import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import TrainingArguments, AutoTokenizer
from trl import DPOTrainer
from datasets import Dataset
from torch import nn
from PIL import Image
import json
from rich import print
from peft import get_peft_model, LoraConfig, TaskType
    
# 冻结 ViT 模块
def freeze_vit(model):
    for name, param in model.named_parameters():
        # 冻结视觉编码模块
        if "vision_tower" in name:
            param.requires_grad = False
        # 冻结 embedding 层
        elif "embed_tokens" in name:
            param.requires_grad = False
        # 冻结 LayerNorm 层（适用于 transformer block 开头/结尾）
        elif "norm" in name or "final_layernorm" in name:
            param.requires_grad = False
    return model

# 加载模型和处理器
def load_model_7b():
    local_model_path = "models/Qwen2.5-VL-3B/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
    local_model_path = "models/Qwen2.5-VL-7B/snapshots/b901af65fa3b2801b73d1c5b1ff59b89d81a708f"
    processor = AutoProcessor.from_pretrained(local_model_path)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        # device_map="auto",
    )
    # model.gradient_checkpointing_enable()
    # 冻结视觉部分
    model = freeze_vit(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # 配置 LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 可根据实际模块名改
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # decoder-only 架构
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数数目

    return model, processor

from typing import Dict, List, Optional, Tuple, Union
from trl.trainer.utils import pad_to_length 

class DPOTrainer_Qwen2_5_VL(DPOTrainer):
    def __init__(
            self,
            processor,
            *args, 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}
        max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        for k in batch:
            # import pdb; pdb.set_trace()
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)
        messages_chosen = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": batch["image_path"],
                    },
                    {
                        "type": "text", 
                        "text": batch["prompt"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text", 
                        "text": batch["chosen_answer"],
                    },
                ],
            }
        ]        
        messages_rejected = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": batch["image_path"],
                    },
                    {
                        "type": "text", 
                        "text": batch["prompt"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text", 
                        "text": batch["rejected_answer"],
                    },
                ],
            }
        ]
        text_chosen = self.processor.apply_chat_template(
            messages_chosen, tokenize=False, add_generation_prompt=True
        )
        text_rejected = self.processor.apply_chat_template(
            messages_rejected, tokenize=False, add_generation_prompt=True
        )
        image_inputs_chosen, video_inputs_chosen = process_vision_info(messages_chosen)
        image_inputs_rejected, video_inputs_rejected = process_vision_info(messages_rejected)
        concatenated_batch["inputs"] = self.processor(
            text=[text_chosen,text_rejected],
            images=[image_inputs_chosen,image_inputs_rejected],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        # 将concatenated_batch["inputs"]["input_ids"][0]和concatenated_batch["labels"]拼接
        concatenated_batch["inputs"]["input_ids"][0]
        return concatenated_batch
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # import pdb; pdb.set_trace()
        concatenated_batch = self.concatenated_inputs(
            batch,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        new_labels = concatenated_batch["concatenated_labels"].clone()
        all_logits = model(
            **concatenated_batch["inputs"],
        ).logits
        # print("[green]all_logits:[/green]", all_logits.shape)
        label_len = max(batch["chosen_labels"][0].shape[0], batch["rejected_labels"][0].shape[0])
        all_logits = all_logits[:, -label_len:, :]
        all_logps = self.get_batch_logps(
            all_logits,
            new_labels,
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = new_labels[:len_chosen]
        rejected_labels = new_labels[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels)











from torch.nn.utils.rnn import pad_sequence

def resize_image_by_scale(image, target_resolution=224*224):
    """按目标分辨率等比例缩小图片"""
    width, height = image.size
    current_res = width * height
    scale_factor = (target_resolution / current_res) ** 0.5  # 缩放因子

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return image.resize((new_width, new_height), Image.LANCZOS)

class VisionDPOCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_labels = []

        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_labels = []

        images = []

        for example in features:
            # 加载图像
            image = Image.open(example["image"]).convert("RGB")
            images.append(image)
            image = resize_image_by_scale(image, target_resolution=1024*1024)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text", 
                            "text": example["prompt"],
                        },
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            chosen_inputs = self.processor(
                text=[text],
                images=[image_inputs],
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            chosen_input_ids.append(chosen_inputs["input_ids"].squeeze(0))
            chosen_attention_mask.append(chosen_inputs["attention_mask"].squeeze(0))
            chosen_label_ids = self.processor.tokenizer(
                example["chosen"], return_tensors="pt", padding=True, truncation=True
            )["input_ids"].squeeze(0)
            chosen_labels.append(chosen_label_ids)
            # print(chosen_labels)
            # print(chosen_labels[-1].shape)  # ✅ 检查 chosen_labels 的形状
            # import pdb; pdb.set_trace()
            # 负样本
            rejected_inputs = chosen_inputs.copy()
            rejected_input_ids.append(rejected_inputs["input_ids"].squeeze(0))
            rejected_attention_mask.append(rejected_inputs["attention_mask"].squeeze(0))
            rejected_label_ids = self.processor.tokenizer(
                example["rejected"], return_tensors="pt", padding=True, truncation=True
            )["input_ids"].squeeze(0)
            rejected_labels.append(rejected_label_ids)

        ret = {
            "image_path":image,
            "prompt": example["prompt"],
            "chosen_answer": example["chosen"],
            "rejected_answer": example["rejected"],
            "chosen_input_ids": pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "chosen_attention_mask": pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0),
            "chosen_labels": pad_sequence(chosen_labels, batch_first=True, padding_value=-100),  # -100 for ignore index
            "rejected_input_ids": pad_sequence(rejected_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "rejected_attention_mask": pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0),
            "rejected_labels": pad_sequence(rejected_labels, batch_first=True, padding_value=-100),
        }
        # print("[blue]VisionDPOCollator: ret", ret)  # ✅ 检查返回的键
        return ret


def main():
    dpo_data_file = "./outputs/dpo_data.json"
    output_dir = "./dpo_finetuned_qwen2.5_vl_7b"

    # 加载模型和处理器
    model, processor = load_model_7b()
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'

    # 加载参考模型（必须传入）
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "models/Qwen2.5-VL-7B/snapshots/b901af65fa3b2801b73d1c5b1ff59b89d81a708f",
        torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    ref_model = freeze_vit(ref_model)
    ref_model.gradient_checkpointing_enable()
    ref_model.config.use_cache = False
    # 加载 DPO 数据集
    dataset = Dataset.from_json(dpo_data_file)

    # 配置训练参数
    dpo_config = TrainingArguments(
        output_dir="./qwen2.5-vl-dpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,   

        fp16=True,                               # 启用混合精度
        bf16=False,                              # 若你显卡支持 bfloat16，可以开启
        torch_compile=False,                     # Torch 2.0 的 compile 有时会爆显存
        optim="adamw_torch",                     # 不使用 fused_adam（占显存多）    
        
        # bf16=True,
        report_to="none",
        remove_unused_columns=False,  # <== 添加这个
    )
    # print(f"model type: {type(model)}")
    # 初始化 Trainer
    trainer = DPOTrainer_Qwen2_5_VL(
        model=model,
        processor=processor,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
        max_target_length=512,
        beta=0.1,
        label_pad_token_id=-100,
        padding_value=tokenizer.pad_token_id,
        truncation_mode="keep_end",
        loss_type="sigmoid",  # 或 "hinge"
        is_encoder_decoder=False,
        data_collator=VisionDPOCollator(processor),
    )
    # print(f"model type: {type(model)}")
    # 训练


    trainer.train()
    trainer.save_model(output_dir)

    print(f"DPO 训练完成，模型已保存至 {output_dir}")

if __name__ == "__main__":
    main()
