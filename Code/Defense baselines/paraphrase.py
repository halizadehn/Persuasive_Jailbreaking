import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PARAPHRASED_PATH = 'paraphrased_results.jsonl'
OUT_PATH = 'paraphrased_vicuna_responses.jsonl'
MODEL_PATH = 'lmsys/vicuna-7b-v1.3'
DEVICE = 'cuda:0'
MAX_NEW_TOKENS = 100

# Load Vicuna model and tokenizer
vicuna_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
vicuna_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE).eval()

def generate_vicuna_response(prompt, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = vicuna_tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        output_ids = vicuna_model.generate(input_ids, max_new_tokens=max_new_tokens)
    response = vicuna_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove prompt from beginning of response if present
    if response.startswith(prompt):
        response = response[len(prompt):].lstrip()
    return response

def main():
    paraphrased_prompts = []
    with open(PARAPHRASED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            paraphrased_prompts.append(json.loads(line))

    # Resume logic: count completed prompts in output file
    start_idx = 0
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"Resuming from prompt {start_idx} (already processed)")

    with open(OUT_PATH, 'a', encoding='utf-8') as f:
        for meta in tqdm(paraphrased_prompts[start_idx:], initial=start_idx, total=len(paraphrased_prompts)):
            paraphrased_prompt = meta.get('paraphrased_prompt')
            if not paraphrased_prompt:
                response = None
            else:
                try:
                    response = generate_vicuna_response(paraphrased_prompt)
                except Exception as e:
                    print(f"Error generating Vicuna response: {e}")
                    response = None
            result = {
                'original_prompt': meta.get('original_prompt'),
                'paraphrased_prompt': paraphrased_prompt,
                'vicuna_response': response
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Saved Vicuna responses for {len(paraphrased_prompts)} paraphrased prompts to {OUT_PATH}")

if __name__ == '__main__':
    main()
