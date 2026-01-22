# ==========================================
# Google Colab Setup Instructions
# ==========================================
# 1. Upload this script and 'train.jsonl' to your Colab environment.
# 2. Install necessary libraries by running this in a cell:
#    !pip install -q -U torch transformers peft bitsandbytes datasets accelerate matplotlib scikit-learn pyyaml
# 3. Run this script:
#    !python train_lora.py
# ==========================================

import os
import argparse
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import cohen_kappa_score
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ---------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------
def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Extract configs for readability
MODEL_CFG = config["model"]
LORA_CFG = config["lora"]
TRAIN_CFG = config["training"]

MODEL_ID = MODEL_CFG["base_model"]
MAX_LENGTH = MODEL_CFG["max_length"]

OUTPUT_DIR = TRAIN_CFG["output_dir"]
TRAIN_FILE = "data/train.jsonl" # Hardcoded or could be in config
NUM_EPOCHS = TRAIN_CFG["num_train_epochs"]
BATCH_SIZE = TRAIN_CFG["per_device_train_batch_size"]
GRADIENT_ACCUMULATION_STEPS = TRAIN_CFG["gradient_accumulation_steps"]
LEARNING_RATE = float(TRAIN_CFG["learning_rate"]) # Ensure float

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Fine-tune SOLAR-10.7B-Instruct with LoRA")
parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset ID (e.g., 'username/dataset'). If None, uses local train.jsonl")
args, _ = parser.parse_known_args()

TRAIN_PATH = "data/converted_train.jsonl"
VAL_PATH = "data/converted_val.jsonl"
OLD_TRAIN_FILE = "data/train.jsonl"

def load_local_jsonl(path):
    if not os.path.exists(path):
        return None
    print(f"Loading data from {path}...")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_chat(example):
    # 1. Handle new format from convert_dataset.py (prompt, text, content, etc.)
    if "prompt" in example and "text" in example:
        instruction = "다음 학술 에세이를 읽고, 평가 기준(내용, 구성, 언어)에 따라 채점한 뒤 결과를 JSON 형식으로 출력하세요."
        question = example["prompt"]
        input_text = example["text"]
        
        # Construct output JSON from numeric scores
        output_dict = {
            "1. 내용": float(example.get("content", 0)),
            "2. 구성": float(example.get("organization", 0)),
            "3. 언어": float(example.get("expression", 0)),
            "총점": float(example.get("holistic", 0))
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)
    
    # 2. Handle standard instruction/input/output format
    else:
        instruction = example.get("instruction", "제시된 에세이를 평가하세요.")
        question = example.get("question", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

    # Construct user message with topic and essay
    topic_str = f"### 주제:\n{question}\n\n" if question else ""
    user_content = f"{instruction}\n\n{topic_str}### 에세이:\n{input_text}"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text}
    ]
    return messages

def get_datasets():
    if args.dataset_name:
        print(f"Loading dataset from Hugging Face Hub: {args.dataset_name}")
        from datasets import load_dataset
        ds_dict = load_dataset(args.dataset_name)
        
        # Determine splits
        train_raw = ds_dict["train"]
        if "validation" in ds_dict:
            test_raw = ds_dict["validation"]
        elif "test" in ds_dict:
            test_raw = ds_dict["test"]
        else:
            print("No validation split found. Splitting 10% from train...")
            split = train_raw.train_test_split(test_size=0.1, seed=42)
            train_raw, test_raw = split["train"], split["test"]
        
        # Apply formatting to HF dataset
        print("Mapping datasets to SOLAR format...")
        train_ds = train_raw.map(lambda x: {"messages": format_chat(x)}, remove_columns=train_raw.column_names)
        test_ds = test_raw.map(lambda x: {"messages": format_chat(x)}, remove_columns=test_raw.column_names)
        
        return train_ds, test_ds
    
    else:
        # Load local files (Fallback)
        train_raw = load_local_jsonl(TRAIN_PATH)
        val_raw = load_local_jsonl(VAL_PATH)
        
        # Fallback to old file if new one doesn't exist
        if train_raw is None:
            print(f"Warning: {TRAIN_PATH} not found. Trying {OLD_TRAIN_FILE}...")
            train_raw = load_local_jsonl(OLD_TRAIN_FILE)
            if train_raw is None:
                raise FileNotFoundError("No training data found in 'data/' folder.")

        # Format and convert to HF Dataset objects
        train_ds = Dataset.from_list([{"messages": format_chat(d)} for d in train_raw])
        
        if val_raw:
            test_ds = Dataset.from_list([{"messages": format_chat(d)} for d in val_raw])
        else:
            print("No validation file found. Splitting 10% from train...")
            split = train_ds.train_test_split(test_size=0.1, seed=42)
            train_ds, test_ds = split["train"], split["test"]
            
        return train_ds, test_ds

train_dataset, test_dataset = get_datasets()
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ---------------------------------------------------------
# 2. Model & Tokenizer (4-bit Quantization)
# ---------------------------------------------------------
print(f"Loading model: {MODEL_ID} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix for fp16 training stability

# Explicitly set chat template for SOLAR if it's missing (Common for this model)
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' + message['content'] + eos_token }}{% endif %}{% endfor %}"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id # Explicitly sync pad_token_id

# Prepare model for k-bit training (gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# ---------------------------------------------------------
# 3. LoRA Configuration
# ---------------------------------------------------------
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_CFG["r"],
    lora_alpha=LORA_CFG["lora_alpha"],
    lora_dropout=LORA_CFG["lora_dropout"],
    target_modules=LORA_CFG["target_modules"],
    bias=LORA_CFG.get("bias", "none")
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ---------------------------------------------------------
# 4. Tokenization
# ---------------------------------------------------------
def preprocess_function(examples):
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in examples["messages"]]
    model_inputs = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = model_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

print("Tokenizing dataset...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# ---------------------------------------------------------
# 5. Training
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=TRAIN_CFG.get("logging_steps", 100),
    save_strategy=TRAIN_CFG.get("save_strategy", "epoch"),
    eval_strategy=TRAIN_CFG.get("evaluation_strategy", "steps"), 
    eval_steps=200,             
    fp16=TRAIN_CFG.get("fp16", True),
    optim=TRAIN_CFG.get("optim", "paged_adamw_8bit"),
    warmup_ratio=TRAIN_CFG.get("warmup_ratio", 0.03),
    group_by_length=TRAIN_CFG.get("group_by_length", True),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print("Starting training...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------
print("Visualizing training loss...")
log_history = trainer.state.log_history

steps = []
losses = []

for log in log_history:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

if steps:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, "training_loss.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
else:
    print("No loss data found in log history.")

# ---------------------------------------------------------
# 7. Statistical Evaluation (QWK Analysis)
# ---------------------------------------------------------
print("\n=== Starting Statistical Evaluation (Base vs. Fine-tuned) ===")

# Select a subset for statistical significance (e.g., 50 samples)
eval_count = min(50, len(test_dataset))
eval_subset = test_dataset.select(range(eval_count))

results = {
    "true_scores": [],
    "base_scores": [],
    "ft_scores": [],
    "details": []
}

def extract_total_score(text):
    """
    Extracts '총점' from the model output. 
    Tries JSON parsing first, then Regex fallback.
    """
    try:
        # Use a non-greedy match to find the first valid JSON-like structure
        match = re.search(r"(.*?)", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Attempt to repair the JSON if it's slightly malformed
            try:
                data = json.loads(json_str)
                return float(data.get("총점", data.get("total_score", -1)))
            except json.JSONDecodeError:
                # Fallback for cases where the model might have added a trailing comma
                clean_json_str = re.sub(r",\s*}}", "}", json_str)
                clean_json_str = re.sub(r",\s*]", "]", clean_json_str)
                data = json.loads(clean_json_str)
                return float(data.get("총점", data.get("total_score", -1)))
    except:
        pass
    
    # Regex Fallback
    try:
        match = re.search(r"""['"]총점['"]\s*:\s*([\d\.]+)""", text)
        if match:
            return float(match.group(1))
    except:
        pass
    return None

print(f"Evaluating {eval_count} samples for QWK accuracy...")

for i, example in enumerate(eval_subset):
    # Extract from formatted messages
    user_prompt_content = example["messages"][0]["content"]
    ground_truth_content = example["messages"][1]["content"]
    
    # Get True Score
    true_score = extract_total_score(ground_truth_content)
    if true_score is None: continue 

    print(f"  [{i+1}/{eval_count}] Ground Truth: {true_score}")
    
    # 1. Base Model Prediction
    with model.disable_adapter():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt_content}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        base_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        base_score = extract_total_score(base_output)

    # 2. Fine-tuned Prediction
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt_content}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    ft_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ft_score = extract_total_score(ft_output)
    
    # Collect Data
    results["true_scores"].append(true_score)
    results["base_scores"].append(base_score if base_score is not None else 0.0)
    results["ft_scores"].append(ft_score if ft_score is not None else 0.0)

    print(f"     -> Base Score: {base_score} | FT Score: {ft_score}")
    
    results["details"].append({
        "true_score": true_score,
        "base_output": base_output,
        "base_score": base_score,
        "ft_output": ft_output,
        "ft_score": ft_score
    })

# --- Metrics Calculation ---

# QWK (Quadratic Weighted Kappa)
def to_int_score(scores):
    return [int(round(max(1.0, min(5.0, s)))) for s in scores]

true_int = to_int_score(results["true_scores"])
base_int = to_int_score(results["base_scores"])
ft_int = to_int_score(results["ft_scores"])

qwk_base = cohen_kappa_score(true_int, base_int, weights='quadratic')
qwk_ft = cohen_kappa_score(true_int, ft_int, weights='quadratic')

print(f"\nResults Summary:")
print(f"  Base Model -> QWK: {qwk_base:.4f}")
print(f"  Fine-tuned -> QWK: {qwk_ft:.4f}")

# Save JSON Results
final_results = {
    "metrics": {
        "qwk_base": qwk_base, "qwk_ft": qwk_ft
    },
    "details": results["details"]
}

with open(os.path.join(OUTPUT_DIR, "statistical_results.json"), "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

# Plot QWK Metrics Comparison
plt.figure(figsize=(8, 6))
models = ['Base', 'Fine-tuned']
qwks = [qwk_base, qwk_ft]
plt.bar(models, qwks, color=['gray', 'orange'])
plt.title('QWK Score Comparison (Higher is Better)')
plt.ylabel('Quadratic Weighted Kappa')
plt.ylim(-0.1, 1.1)
for i, v in enumerate(qwks):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom')

plot_path = os.path.join(OUTPUT_DIR, "qwk_comparison.png")
plt.savefig(plot_path)
print(f"QWK Metrics Plot saved to {plot_path}")

print("Done! Download the folder to check 'qwk_comparison.png'.")