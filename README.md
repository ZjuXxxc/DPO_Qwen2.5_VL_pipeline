# 🔍 DPO_Qwen2.5_VL_Pipeline

A **Weak-to-Strong Learning Pipeline** for enhancing **Qwen2.5-VL-7B** in image captioning tasks through **Direct Preference Optimization (DPO)**, significantly reducing hallucinations and improving caption quality.

## 📌 Overview

Vision-Language Models (VLMs) often suffer from hallucination and incomplete descriptions. This project introduces a **weak-to-strong training pipeline** leveraging a smaller model **Qwen2.5-VL-3B** to generate and validate preference data, which guides the larger model **Qwen2.5-VL-7B** using DPO fine-tuning.

We demonstrate the effectiveness of this pipeline on the **COCO2017** dataset, and evaluate using **MMHal-Bench**, **AMBER**, and **LLaVA-Bench-in-the-Wild**.

## 🏗️ Pipeline Structure

```text
1_prepare_images.py              # Prepare the images
2_generate_descriptions.py       # Use Qwen2.5-VL-7B to generate sentence variants
3_regenerate_sentences.py        # Qwen2.5-VL-7B regenerate the variants
4_validate_sentences.py          # Qwen2.5-VL-3B validates and ranks the variants
5_build_dpo_dataset.py           # Construct DPO preference pairs
6_train_dpo.py                   # DPO training using LoRA + DeepSpeed
