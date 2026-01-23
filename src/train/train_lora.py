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
from datasets import Dataset, concatenate_datasets, load_dataset
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
parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset ID (e.g., 'username/dataset').")
parser.add_argument("--experiment_mode", type=str, default="full", 
                    choices=["zero_shot", "original", "augmented", "full"],
                    help="Select experiment mode: zero_shot (no train), original (orig only), augmented (orig+aug), full (orig+aug+gen)")
args, _ = parser.parse_known_args()

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

# Dynamic Output Directory based on Experiment Mode
OUTPUT_DIR = f"{TRAIN_CFG['output_dir']}_{args.experiment_mode}"
NUM_EPOCHS = TRAIN_CFG["num_train_epochs"]
BATCH_SIZE = TRAIN_CFG["per_device_train_batch_size"]
GRADIENT_ACCUMULATION_STEPS = TRAIN_CFG["gradient_accumulation_steps"]
LEARNING_RATE = float(TRAIN_CFG["learning_rate"])

TRAIN_PATH = "data/converted_train.jsonl"
VAL_PATH = "data/converted_val.jsonl"

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
    dataset_name = args.dataset_name if args.dataset_name else "SJunha/aes-dataset"
    print(f"Loading datasets from Hugging Face Hub: {dataset_name}")
    print(f"Experiment Mode: {args.experiment_mode}")
    
    # 1. Load Original Data (Always needed for Test, and mostly for Train)
    print("Loading Original Data (Train/Val)...")
    try:
        original_ds = load_dataset(dataset_name, data_files={"train": "train.jsonl", "test": "validation.jsonl"})
        train_ds_1 = original_ds["train"]
        test_ds = original_ds["test"]
    except Exception as e:
        print(f"Error loading original data: {e}")
        raise e

    if args.experiment_mode == "zero_shot":
        # Only process test set
        print("Zero-shot mode: Skipping training data loading.")
        test_ds = test_ds.map(lambda x: {"messages": format_chat(x)}, remove_columns=test_ds.column_names)
        return None, test_ds

    processed_train_datasets = []

    # A. Original Data (Included in original, augmented, full)
    ds1_mapped = train_ds_1.map(lambda x: {"messages": format_chat(x)}, remove_columns=train_ds_1.column_names)
    processed_train_datasets.append(ds1_mapped)
    
    # B. Augmented Data (Included in augmented, full)
    if args.experiment_mode in ["augmented", "full"]:
        print("Loading Augmented Data (Gold Augmented)...")
        try:
            augmented_ds_dict = load_dataset(dataset_name, "gold_augmented")
            ds3_mapped = augmented_ds_dict["train"].map(lambda x: {"messages": format_chat(x)}, remove_columns=augmented_ds_dict["train"].column_names)
            processed_train_datasets.append(ds3_mapped)
        except Exception as e:
            print(f"Warning: Could not load augmented data: {e}")

    # C. Generated Data (Included in full only)
    if args.experiment_mode == "full":
        print("Loading Generated Data (PaperClinic)...")
        try:
            generated_ds_dict = load_dataset(dataset_name, data_files={"train": "paperclinic_generated_dataset.jsonl"})
            ds2_mapped = generated_ds_dict["train"].map(lambda x: {"messages": format_chat(x)}, remove_columns=generated_ds_dict["train"].column_names)
            processed_train_datasets.append(ds2_mapped)
        except Exception as e:
            print(f"Warning: Could not load generated data: {e}")

    # Process Test
    test_ds = test_ds.map(lambda x: {"messages": format_chat(x)}, remove_columns=test_ds.column_names)

    # Concatenate
    print(f"Concatenating {len(processed_train_datasets)} training datasets...")
    full_train_ds = concatenate_datasets(processed_train_datasets)
    
    return full_train_ds, test_ds

train_dataset, test_dataset = get_datasets()
if train_dataset:
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
else:
    print(f"Test samples: {len(test_dataset)} (Zero-shot mode)")

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
tokenizer.padding_side = "right" 

# Explicitly set chat template for SOLAR if it's missing
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' + message['content'] + eos_token }}{% endif %}{% endfor %}"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id 

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ---------------------------------------------------------
# 3. LoRA Configuration & Training
# ---------------------------------------------------------

if args.experiment_mode != "zero_shot":
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
    
    # Save Loss Plot
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
        plt.title(f"Training Loss ({args.experiment_mode})")
        plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))

else:
    print("Zero-shot mode: Skipping training process.")
    # For zero-shot, we still need to setup 'model' for inference, which is already done.
    # We just don't wrap it in a Trainer or apply LoRA training config (or apply it but don't train).
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# ---------------------------------------------------------
# 7. Statistical Evaluation (QWK Analysis)
# ---------------------------------------------------------
print(f"\n=== Starting Statistical Evaluation (Mode: {args.experiment_mode}) ===")

# Select a subset for statistical significance
eval_count = min(50, len(test_dataset))
eval_subset = test_dataset.select(range(eval_count))

results = {
    "true_scores": [],
    "pred_scores": [],
    "details": []
}

def extract_total_score(text):
    try:
        match = re.search(r"(.*?)", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return float(data.get("총점", data.get("total_score", -1)))
            except:
                pass
    except:
        pass
    try:
        match = re.search(r"""['"]총점['"]\s*:\s*([\d\.]+)""", text)
        if match: return float(match.group(1))
    except: pass
    return None

print(f"Evaluating {eval_count} samples for QWK accuracy...")

# Ensure model is in eval mode
model.eval()

for i, example in enumerate(eval_subset):
    user_prompt_content = example["messages"][0]["content"]
    ground_truth_content = example["messages"][1]["content"]
    
    true_score = extract_total_score(ground_truth_content)
    if true_score is None: continue 

    print(f"  [{i+1}/{eval_count}] Ground Truth: {true_score}")
    
    # Prediction
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
    pred_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred_score = extract_total_score(pred_output)
    
    results["true_scores"].append(true_score)
    results["pred_scores"].append(pred_score if pred_score is not None else 0.0)

    print(f"     -> Pred Score: {pred_score}")
    
    results["details"].append({
        "true_score": true_score,
        "pred_output": pred_output,
        "pred_score": pred_score
    })

# --- Metrics Calculation ---
def to_int_score(scores):
    return [int(round(max(1.0, min(5.0, s)))) for s in scores]

true_int = to_int_score(results["true_scores"])
pred_int = to_int_score(results["pred_scores"])

qwk = cohen_kappa_score(true_int, pred_int, weights='quadratic')

print(f"\nResults Summary ({args.experiment_mode}):")
print(f"  QWK: {qwk:.4f}")

# Save JSON Results
final_results = {
    "mode": args.experiment_mode,
    "metrics": {"qwk": qwk},
    "details": results["details"]
}

with open(os.path.join(OUTPUT_DIR, "statistical_results.json"), "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

print(f"Evaluation complete. Results saved to {OUTPUT_DIR}")