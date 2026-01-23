# ==========================================
# Google Colab Setup Instructions
# ==========================================
# 1. Open this file in Google Colab.
# 2. Run the installation cell below.
# 3. Add your Hugging Face Token in the secrets tab (Name: HF_TOKEN) or paste it when prompted.
# 4. Run the main experiment.
# ==========================================

# --- [Cell 1] Installation ---
# !pip install -q -U torch transformers peft bitsandbytes datasets accelerate matplotlib scikit-learn huggingface_hub

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import cohen_kappa_score
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from huggingface_hub import login

# ==========================================
# Configuration
# ==========================================
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"  # Verify exact ID on HF
DATASET_ID = "SJunha/aes-dataset"
OUTPUT_DIR = "aes_pilot_results"

# Training Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1  # Pilot experiment: keep it short
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
MAX_LENGTH = 1024

# Ensure reproducible runs
SEED = 42
torch.manual_seed(SEED)

# ==========================================
# 1. Setup & Data Loading
# ==========================================
def setup_environment():
    # Login to HF if needed
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    else:
        try:
            from google.colab import userdata
            login(token=userdata.get('HF_TOKEN'))
        except:
            print("HF_TOKEN not found. Please login manually if datasets are private.")

def format_chat(example):
    """
    Formats the raw dataset example into a chat structure.
    Expects input features: 'instruction', 'input' (essay), 'output' (score JSON).
    """
    # Adjust field names based on your actual dataset schema
    # Case A: Combined/Augmented format
    if "instruction" in example and "input" in example:
        user_content = f"{example['instruction']}\n\n### 에세이:\n{example['input']}"
        assistant_content = example["output"]
    # Case B: Original raw format (fallback)
    elif "prompt" in example and "text" in example:
        # Construct JSON output from individual scores
        scores = {
            "1. 내용": example.get("content", 0),
            "2. 구성": example.get("organization", 0),
            "3. 언어": example.get("expression", 0),
            "총점": example.get("holistic", 0)
        }
        user_content = f"다음 학술 에세이를 읽고, 평가 기준(내용, 구성, 언어)에 따라 채점한 뒤 결과를 JSON 형식으로 출력하세요.\n\n### 에세이:\n{example['text']}"
        assistant_content = json.dumps(scores, ensure_ascii=False)
    else:
        # Generic fallback
        user_content = str(example)
        assistant_content = ""

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    return messages

def load_data():
    print("Loading datasets from Hugging Face...")
    
    # 1. Original Data
    # Assuming 'data/converted_train.jsonl' was uploaded to 'train' split default config
    try:
        ds_original = load_dataset(DATASET_ID, split="train")
    except:
        print(f"Could not load default config for {DATASET_ID}. Checking structure...")
        # Fallback if multiple configs exist
        ds_original = load_dataset(DATASET_ID, data_files="train.jsonl", split="train")

    # 2. Augmented Data
    try:
        ds_augmented = load_dataset(DATASET_ID, "gold_augmented", split="train")
    except Exception as e:
        print(f"Warning: Augmented config not found ({e}). Using only original for now.")
        ds_augmented = None

    # 3. Validation Data
    try:
        ds_val = load_dataset(DATASET_ID, data_files="validation.jsonl", split="train") # Check split name if uploaded manually
    except:
        # If validation file isn't found, split from original
        print("Validation file not found. Splitting from original...")
        split = ds_original.train_test_split(test_size=0.1, seed=SEED)
        ds_original = split["train"]
        ds_val = split["test"]

    # --- Normalize Schemas for Concatenation ---
    def normalize_original(ex):
        scores = {
            "1. 내용": ex.get("content", 0),
            "2. 구성": ex.get("organization", 0),
            "3. 언어": ex.get("expression", 0),
            "총점": ex.get("holistic", 0)
        }
        return {
            "instruction": "다음 학술 에세이를 읽고, 평가 기준(내용, 구성, 언어)에 따라 채점한 뒤 결과를 JSON 형식으로 출력하세요.",
            "input": ex["text"],
            "output": json.dumps(scores, ensure_ascii=False)
        }

    # Normalize Original
    if "text" in ds_original.column_names:
        print("Normalizing Original Dataset schema...")
        ds_original = ds_original.map(normalize_original, remove_columns=ds_original.column_names)
        
    # Normalize Validation (for consistency, though less critical for eval if format_chat handles it)
    if "text" in ds_val.column_names:
        ds_val = ds_val.map(normalize_original, remove_columns=ds_val.column_names)

    # Align Augmented (ensure it has only these columns or enough to match)
    # Augmented already has instruction, input, output. We might need to remove 'question' if it exists.
    if ds_augmented:
        keep_cols = ["instruction", "input", "output"]
        ds_augmented = ds_augmented.select_columns([c for c in keep_cols if c in ds_augmented.column_names])

    print(f"Original Train: {len(ds_original)}")
    print(f"Augmented Train: {len(ds_augmented) if ds_augmented else 0}")
    print(f"Validation: {len(ds_val)}")

    return ds_original, ds_augmented, ds_val

# ==========================================
# 2. Evaluation Logic (QWK)
# ==========================================
def extract_total_score(text):
    try:
        match = re.search(r"['"]?총점['"]?\s*:\s*([\d\.]+)", text)
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.0

def evaluate_qwk(model, tokenizer, dataset, sample_size=50):
    """
    Generates responses for a subset of the dataset and calculates QWK.
    """
    model.eval()
    subset = dataset.select(range(min(len(dataset), sample_size)))
    
    true_scores = []
    pred_scores = []
    
    print(f"Running evaluation on {len(subset)} samples...")
    
    for i, example in enumerate(subset):
        # Prepare Input
        msgs = format_chat(example)
        user_msg = msgs[0]["content"]
        true_resp = msgs[1]["content"]
        
        # Get Ground Truth
        true_val = extract_total_score(true_resp)
        true_scores.append(int(round(true_val)))
        
        # Generate
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_val = extract_total_score(gen_text)
        pred_scores.append(int(round(max(1.0, min(5.0, pred_val)))))
        
        if i % 10 == 0:
            print(f"  Sample {i}: True={true_val} | Pred={pred_val}")

    # Calculate QWK
    qwk = cohen_kappa_score(true_scores, pred_scores, weights='quadratic')
    print(f"  >> QWK Score: {qwk:.4f}")
    return qwk

# ==========================================
# 3. Training Logic
# ==========================================
def get_model_and_tokenizer():
    print(f"Loading base model: {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Ensure chat template exists
    if tokenizer.chat_template is None:
         tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' + message['content'] + eos_token }}{% endif %}{% endfor %}"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def train_condition(condition_name, train_ds, eval_ds):
    print(f"\n\n=== Starting Experiment: {condition_name} ===")
    
    model, tokenizer = get_model_and_tokenizer()
    
    # Setup LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Adjust for LFM if needed
    )
    model = get_peft_model(model, peft_config)
    
    # Preprocess
    def preprocess(examples):
        texts = [
            tokenizer.apply_chat_template(format_chat(ex), tokenize=False) 
            for ex in zip(examples['instruction'], examples['input'], examples['output']) 
            if 'instruction' in examples # Handle different formats roughly
        ]
        # Robust formatting map
        formatted_texts = []
        for i in range(len(examples['input'])):
            # Reconstruct example dict
            ex = {k: examples[k][i] for k in examples}
            msgs = format_chat(ex)
            formatted_texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            
        model_inputs = tokenizer(
            formatted_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = model_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    print("Tokenizing...")
    train_tokenized = train_ds.map(preprocess, batched=True)
    
    # Trainer
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{condition_name}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    print(f"Training for {NUM_EPOCHS} epochs...")
    trainer.train()
    
    print("Evaluating...")
    qwk = evaluate_qwk(model, tokenizer, eval_ds)
    
    # Cleanup to save VRAM
    del model, trainer
    torch.cuda.empty_cache()
    
    return qwk

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    setup_environment()
    ds_original, ds_augmented, ds_val = load_data()
    
    results = {}
    
    # --- Condition 1: Not Trained (Zero-shot) ---
    print("\n\n=== Condition 1: Not Trained (Base Model) ===")
    model, tokenizer = get_model_and_tokenizer()
    results["Not Trained"] = evaluate_qwk(model, tokenizer, ds_val)
    del model
    torch.cuda.empty_cache()
    
    # --- Condition 2: Tuned with Original ---
    results["Original Only"] = train_condition("original_only", ds_original, ds_val)
    
    # --- Condition 3: Tuned with Aug + Original ---
    if ds_augmented:
        ds_combined = concatenate_datasets([ds_original, ds_augmented])
        # Shuffle
        ds_combined = ds_combined.shuffle(seed=SEED)
        results["Aug + Original"] = train_condition("aug_original", ds_combined, ds_val)
    else:
        print("Skipping Condition 3 (No augmented data).")
    
    # --- Report ---
    print("\n\n==========================================")
    print("       FINAL PILOT RESULTS (QWK)          ")
    print("==========================================")
    for name, score in results.items():
        print(f"{name:20s}: {score:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['gray', 'blue', 'green'])
    plt.title("Pilot Experiment: Data Augmentation Effect")
    plt.ylabel("QWK Score")
    plt.ylim(0, 1.0)
    plt.savefig(f"{OUTPUT_DIR}/pilot_results.png")
    print(f"Plot saved to {OUTPUT_DIR}/pilot_results.png")

if __name__ == "__main__":
    main()
