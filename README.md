## Fine-tuning-Qwen2.5
This repository demonstrates how to fine‑tune the Qwen/Qwen2.5‑1.5B language model using QLoRA, enabling efficient training on consumer GPUs while achieving strong performance on custom instruction‑based tasks.
The project includes:
- A complete QLoRA fine‑tuning pipeline
- Dataset formatting guidelines
- Evaluation scripts
- Instructions for adapting the pipeline to any domain
This setup is ideal for anyone who wants to build a domain‑specific AI assistant with custom tone, behavior, and knowledge.

## 🚀 Features
- Fine‑tuning using 4‑bit QLoRA
- Works on GPUs as small as 12–24 GB
- Instruction‑style supervised fine‑tuning (SFT)
- LoRA adapters applied to attention + MLP layers
- Evaluation against the original base model
- Easy to adapt for any dataset

## 🧩 Why Fine‑Tune Qwen?
Qwen 2.5 models are strong general‑purpose LLMs, but they are not optimized for specialized domains.
**Fine‑tuning allows you to:**
- Add domain‑specific knowledge
- Customize tone and style
- Improve accuracy on niche tasks
- Reduce hallucinations
- Enforce custom safety rules
- Build predictable, consistent behavior
This makes Qwen 2.5 (1.5B) a great foundation for lightweight, specialized AI systems.

## 📦 Dataset Format
**Your dataset must be in JSONL format:**
```bash
    {"instruction": "What is photosynthesis?", "output": "It is how plants make food using sunlight."}
    {"instruction": "Explain gravity simply.", "output": "Gravity pulls things toward the ground."}
```
**Required fields:**
- instruction → user question
- output → model answer

**Optional fields:**
- category
- metadata

## 🛠️ Fine‑Tuning Pipeline

**1. Load dataset**
Using datasets.load_dataset to read JSONL files.

**2. Load Qwen base model**
Loaded in 4‑bit quantized mode using BitsAndBytes.

**3. Prepare for QLoRA**
prepare_model_for_kbit_training() stabilizes training.

**4. Apply LoRA adapters**
Adapters are injected into:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
These layers control attention and MLP behavior.

**5. Train with SFTTrainer**
Supervised fine‑tuning on your instruction → output pairs.

**6. Evaluate**
Compare original vs fine‑tuned model on test.jsonl

## 🧰 How to Fine‑Tune on Your Own Dataset
**1. Prepare your dataset**
Create a JSONL file:
```bash
{"instruction": "...", "output": "..."}
```

**2. Update file paths**
In the training script:
```bash
train_data = load_dataset("json", data_files="train.jsonl")
```

**3. Adjust formatting function**
Example:
```bash
def format_example(e):
    return f"Instruction: {e['instruction']}\nAnswer: {e['output']}"
```

**4. Run training**
Use the provided QLoRA script.

**5. Evaluate**
Run the evaluation script to compare performance.

## 🧑‍💻 Example: Running Inference
```bash
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

prompt = "Instruction: Explain gravity simply.\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## 🙌 Acknowledgements
- Qwen team for the base model
- HuggingFace Transformers
- TRL (SFTTrainer)
- PEFT (LoRA)
- BitsAndBytes (4‑bit quantization)

## 👨‍💻 Author
Mohammed Abdul Bari

⭐ Star this repo if it helps your social media game!






