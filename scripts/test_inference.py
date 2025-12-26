from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Paths to the saved model
model_path = "../outputs/finetuned-model"
base_model_name = "microsoft/phi-2"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the PEFT adapters
model = PeftModel.from_pretrained(base_model, model_path)

print("Fine-tuned model loaded for inference test.")

# Test with a sample prompt
test_prompt = "You are a helpful AI assistant.\nHuman: How do I write a Python function to calculate the factorial of a number?\nAssistant:"
print(f"Test Prompt: {test_prompt}")

# Tokenize
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        min_length=inputs.input_ids.shape[1] + 10,  # Ensure at least 10 new tokens
        max_new_tokens=200,  # Generate up to 200 new tokens
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )

# Decode the full generated text
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Full Generated Response: {full_response}")

# Extract only the assistant's response
response_start = full_response.find("Assistant:")
if response_start != -1:
    response = full_response[response_start + len("Assistant:"):].strip()
else:
    response = full_response[len(test_prompt):].strip()

print(f"Extracted Assistant Response: {response}")
