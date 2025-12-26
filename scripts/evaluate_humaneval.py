import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness
import json

def load_base_model(model_name="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def load_fine_tuned_model(model_path="../outputs/finetuned-model", base_model_name="microsoft/phi-2"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    return model, tokenizer

def generate_code(model, tokenizer, prompt, max_length=512, temperature=0.2, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length - inputs.input_ids.shape[1],
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return prompt + generated

def evaluate_model(model_type="base"):
    if model_type == "base":
        model, tokenizer = load_base_model()
    elif model_type == "finetuned":
        model, tokenizer = load_fine_tuned_model()
    else:
        raise ValueError("model_type must be 'base' or 'finetuned'")

    problems = read_problems()
    samples = []
    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        # Generate multiple completions for accuracy (pass@k)
        for i in range(5):  # Generate 5 to compute pass@5
            code = generate_code(model, tokenizer, prompt)
            samples.append({
                "task_id": task_id,
                "completion": code[len(prompt):]  # Only the generated part
            })

    # Write samples to jsonl for evaluation
    os.makedirs("../outputs/humaneval_results", exist_ok=True)
    sample_file = f"../outputs/humaneval_results/samples_{model_type}.jsonl"
    with open(sample_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    # Evaluate
    results = evaluate_functional_correctness(sample_file, timeout=10.0)
    pass_at_1 = results["pass@1"]
    pass_at_k = {}
    for k in [2, 5]:
        if f"pass@{k}" in results:
            pass_at_k[k] = results[f"pass@{k}"]

    return pass_at_1, pass_at_k

if __name__ == "__main__":
    model_types = ["base", "finetuned"]
    results = {}
    for mt in model_types:
        print(f"Evaluating {mt} model...")
        try:
            pass1, passk = evaluate_model(mt)
            results[mt] = {"pass@1": pass1, "pass@k": passk}
            print(f"{mt} pass@1: {pass1}")
            print(f"{mt} pass@k: {passk}")
        except Exception as e:
            print(f"Error evaluating {mt}: {e}")
            results[mt] = {"error": str(e)}

    # Save results
    with open("../outputs/humaneval_results/eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Evaluation completed. Results saved.")
