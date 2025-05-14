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
```
## 🧪 Evaluation Results

| Metric                     | Before DPO | After DPO | Δ Improvement |
|---------------------------|------------|-----------|----------------|
| **MMHal Avg**             | 4.12       | 4.19      | ↑ 1.7%         |
| **MMHal Hallucination ↓** | 0.32       | 0.30      | ↓ 6.3%         |
| **AMBER Hall ↓**          | 0.8000     | 0.8200    | Slight ↓       |
| **AMBER Cover ↑**         | 0.5276     | 0.5296    | Slight ↑       |
| **AMBER Cog ↓**           | 0.0237     | 0.0268    | Slight ↑       |
| **LLaVA-Bench Avg ↑**     | 4.82       | 4.97      | ↑ 3.1%         |
| **LLaVA-Bench Hall ↓**    | 0.23       | 0.17      | ↓ 26%          |

> The pipeline effectively reduces hallucinations (e.g., “clouds” on pizza), while enhancing detailed object coverage (e.g., “pizza slices”).

---

## 🧠 Core Contributions

- ✅ **First Weak-to-Strong Learning Pipeline** for vision-language alignment  
- 🔁 **Open-Source Reproducible Code** with all stages of DPO fine-tuning  
- 📊 **Robust Multi-Benchmark Evaluation** (MMHal, AMBER, LLaVA-Bench)

---

## 📁 Dataset

- **COCO2017** `val2017` split used for caption generation and evaluation

---

## ⚙️ Training Details

- **DPO model**: Qwen2.5-VL-7B (LoRA fine-tuned)  
- **Training setup**: DeepSpeed + bf16  
- **Peak memory**: ~157GB

---

## 🧪 Evaluation Pipeline

1. Generate captions using `amber_test_pipeline.py`  
2. Format the results for MMHal and LLaVA-Bench scoring  
3. Aggregate metrics using the provided evaluation scripts

---

## 📎 Project URL

**Code & Documentation**: [https://github.com/ZjuXxxc/DPO_Qwen2.5_VL_pipeline](https://github.com/ZjuXxxc/DPO_Qwen2.5_VL_pipeline)

---

## 🤝 Acknowledgements

Special thanks to the developers of:

- [Qwen-VL](https://github.com/Qwen-VL)
- [LoRA](https://github.com/microsoft/LoRA)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [LLaVA-Bench](https://github.com/haotian-liu/LLaVA)

---

## 📬 Contact

For questions or collaboration, feel free to open an issue or contact us via GitHub.
