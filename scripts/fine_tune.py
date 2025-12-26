from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

model_name = "microsoft/phi-2"  # Fine-tuning on CodeAlpaca for code instruction following
dataset_name = "sahil2801/CodeAlpaca-20k"

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Load model in 8-bit for memory efficiency
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model with quantization and on GPU
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
print("Quantized model loaded on GPU.")

# Attach trainable adapters using PEFT LoRA 
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2", "dense"]
)
model = get_peft_model(model, peft_config)
print("PEFT adapters attached successfully for fine-tuning.")
dataset = load_dataset(dataset_name)

def preprocess(example):
    if example["input"].strip():
        full_prompt = f"### Instruction: {example['instruction']}\n\n### Input: {example['input']}\n\n### Response: {example['output']}"
    else:
        full_prompt = f"### Instruction: {example['instruction']}\n\n### Response: {example['output']}"

    tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# CodeAlpaca has a 'train' split, preprocess it
full_data = dataset["train"].map(preprocess)
split_datasets = full_data.train_test_split(test_size=0.1, seed=42)
train_data = split_datasets["train"]
eval_data = split_datasets["test"]

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="../outputs/finetuned-model",
    per_device_train_batch_size=2,
    num_train_epochs=1,  # Start with 1 epoch for testing
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()

model.save_pretrained("../outputs/finetuned-model")
tokenizer.save_pretrained("../outputs/finetuned-model")
