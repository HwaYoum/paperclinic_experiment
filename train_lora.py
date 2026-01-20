# ==========================================
# Google Colab Setup Instructions
# ==========================================
# 1. Upload this script and 'train.jsonl' to your Colab environment.
# 2. Install necessary libraries by running this in a cell:
#    !pip install -q -U torch transformers peft bitsandbytes datasets accelerate matplotlib scikit-learn
# 3. Run this script:
#    !python train_lora.py
# ==========================================

import os
import argparse
import json
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
# Argument Parsing
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Fine-tune SOLAR-10.7B-Instruct with LoRA")
parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset ID (e.g., 'username/dataset'). If None, uses local train.jsonl")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Using Upstage SOLAR-10.7B-Instruct.
# Note: This model requires approx 12GB+ VRAM with 4-bit quantization.
MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0" 
OUTPUT_DIR = "aes_finetuned"
TRAIN_FILE = "train.jsonl"
NUM_EPOCHS = 3 # Increased to 3 for better convergence with larger batch size
BATCH_SIZE = 16 
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = 16 * 4 = 64
LEARNING_RATE = 2e-4

# ---------------------------------------------------------
# 1. Load & Format Data
# ---------------------------------------------------------
def load_data():
    if args.dataset_name:
        print(f"Loading dataset from Hugging Face Hub: {args.dataset_name}")
        from datasets import load_dataset
        # Load from Hub
        ds = load_dataset(args.dataset_name, split="train")
        return ds
    else:
        print(f"Loading data from local file: {TRAIN_FILE}...")
        data = []
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

def format_chat(example):
    # Determine if input is a dict (local load) or HF dataset row
    # HF dataset row keys are directly accessible
    
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    # SOLAR format: Merge system instruction into the user message for better compatibility
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": output_text}
    ]
    return messages

raw_data = load_data()

# Handle different data types (List[dict] vs Dataset)
if isinstance(raw_data, list):
    formatted_data = [format_chat(d) for d in raw_data]
    hf_dataset = Dataset.from_list([{"messages": msgs} for msgs in formatted_data])
else:
    # It's a Hugging Face Dataset object
    hf_dataset = raw_data.map(lambda x: {"messages": format_chat(x)}, remove_columns=raw_data.column_names)

# Split: 90% Train, 10% Test
print("Splitting data into Train (90%) and Test (10%)...")
split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

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
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\\n' + message['content'] + eos_token }}{% endif %}{% endfor %}"

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
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
        max_length=1024,
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
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="steps", # Evaluate during training
    eval_steps=200,             # Evaluate every 200 steps
    fp16=True, 
    optim="paged_adamw_8bit",
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
    "base_errors": [],
    "ft_errors": [],
    "details": []
}

def extract_total_score(text):
    """
    Extracts '총점' from the model output. 
    Tries JSON parsing first, then Regex fallback.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return float(data.get("총점", data.get("total_score", -1)))
    except:
        pass
    
    # Regex Fallback
    try:
        match = re.search(r"['\"]총점['\"]\s*:\s*([\d\.]+)", text)
        if match:
            return float(match.group(1))
    except:
        pass
    return None

def generate_response(model, tokenizer, instruction, input_text):
    # Merge instruction into user message for SOLAR compatibility
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"}
    ]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False 
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

print(f"Evaluating {eval_count} samples for QWK accuracy...")

for i, example in enumerate(eval_subset):
    instruction = example["messages"][0]["content"]
    input_text = example["messages"][1]["content"]
    ground_truth_text = example["messages"][2]["content"]
    
    # Get True Score
    true_score = extract_total_score(ground_truth_text)
    if true_score is None: continue 

    print(f"  [{i+1}/{eval_count}] Ground Truth: {true_score}")
    
    # 1. Base Model Prediction
    with model.disable_adapter():
        base_output = generate_response(model, tokenizer, instruction, input_text)
    base_score = extract_total_score(base_output)
    
    # 2. Fine-tuned Prediction
    ft_output = generate_response(model, tokenizer, instruction, input_text)
    ft_score = extract_total_score(ft_output)
    
    # Collect Data
    results["true_scores"].append(true_score)
    
    if base_score is not None:
        results["base_scores"].append(base_score)
        results["base_errors"].append((base_score - true_score)**2)
    else:
        results["base_scores"].append(0.0)
        results["base_errors"].append((0.0 - true_score)**2)

    if ft_score is not None:
        results["ft_scores"].append(ft_score)
        results["ft_errors"].append((ft_score - true_score)**2)
    else:
        results["ft_scores"].append(0.0)
        results["ft_errors"].append((0.0 - true_score)**2)

    print(f"     -> Base Score: {base_score} | FT Score: {ft_score}")
    
    results["details"].append({
        "true_score": true_score,
        "base_output": base_output,
        "base_score": base_score,
        "ft_output": ft_output,
        "ft_score": ft_score
    })

# --- Metrics Calculation ---

# 1. MSE (Mean Squared Error)
mse_base = np.mean(results["base_errors"]) if results["base_errors"] else 0
mse_ft = np.mean(results["ft_errors"]) if results["ft_errors"] else 0

# 2. QWK (Quadratic Weighted Kappa)
def to_int_score(scores):
    return [int(round(max(1.0, min(5.0, s)))) for s in scores]

true_int = to_int_score(results["true_scores"])
base_int = to_int_score(results["base_scores"])
ft_int = to_int_score(results["ft_scores"])

qwk_base = cohen_kappa_score(true_int, base_int, weights='quadratic')
qwk_ft = cohen_kappa_score(true_int, ft_int, weights='quadratic')

print(f"\nResults Summary:")
print(f"  Base Model -> MSE: {mse_base:.4f}, QWK: {qwk_base:.4f}")
print(f"  Fine-tuned -> MSE: {mse_ft:.4f}, QWK: {qwk_ft:.4f}")

# Save JSON Results
final_results = {
    "metrics": {
        "mse_base": mse_base, "mse_ft": mse_ft,
        "qwk_base": qwk_base, "qwk_ft": qwk_ft
    },
    "details": results["details"]
}

with open(os.path.join(OUTPUT_DIR, "statistical_results.json"), "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

# Plot Metrics Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# MSE Plot
models = ['Base', 'Fine-tuned']
mses = [mse_base, mse_ft]
ax1.bar(models, mses, color=['gray', 'skyblue'])
ax1.set_title('MSE (Lower is Better)')
ax1.set_ylabel('Mean Squared Error')
for i, v in enumerate(mses):
    ax1.text(i, v, f"{v:.4f}", ha='center', va='bottom')

# QWK Plot
qwks = [qwk_base, qwk_ft]
ax2.bar(models, qwks, color=['gray', 'orange'])
ax2.set_title('QWK (Higher is Better)')
ax2.set_ylabel('Quadratic Weighted Kappa')
ax2.set_ylim(-0.1, 1.1)
for i, v in enumerate(qwks):
    ax2.text(i, v, f"{v:.4f}", ha='center', va='bottom')

plot_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
plt.savefig(plot_path)
print(f"Metrics Plot saved to {plot_path}")

print("Done! Download the folder to check 'metrics_comparison.png'.")
