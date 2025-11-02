import numpy as np
import ast
import csv
import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
import os
from strong_reject.evaluate import evaluate
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def sentence_ppl(sentence: str, model_name: str = "gpt2") -> float:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    return perplexity


def get_PPL(prompt_file):
    df = pd.read_csv(prompt_file, encoding="utf-8")
    print(f"Total prompts to evaluate: {len(df)}")
    ppl = 0
    for i in range(len(df)):
        prompt = df.loc[i, "jailbreak_prompt"]
        p = calculate_sentence_ppl(prompt, "gpt2")
        ppl += p
    return ppl/len(df)


def calculate_ASR(response_file, ASR_score_file):
    df = pd.read_csv(response_file, encoding="utf-8")
    new_df = pd.DataFrame()
    for i in range(len(df)):
        forbidden_prompt = df.loc[i, "prompt"]
        response = str(df.loc[i,"response"])
        result = evaluate(forbidden_prompt, response, ["string_matching"])
        new_df.loc[i,"prompt_number"] = str(i)
        new_df.loc[i, "evaluation_result"] = str(result[0].get("score"))
    new_df.to_csv(ASR_score_file, index=None)


def get_ASR(ASR_score_file):
    df = pd.read_csv(ASR_score_file)
    scores = df["evaluation_result"].tolist()
    return scores.count(1) / len(df)


def calculate_informative_scores(response_file, IS_score_file):
        df = pd.read_csv(response_file, encoding="utf-8")
        print(f"Total prompts to evaluate: {len(df)}")
        dataset = Dataset.from_pandas(df[["original_prompt", "response"]].rename(columns={"original_prompt": "forbidden_prompt"}))
        results = evaluate_dataset(dataset, evaluators=["strongreject_finetuned"])
        with open(IS_score_file, mode='w', newline='', encoding='utf-8') as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(["prompt_number", "evaluation_result"])
            for prompt_number, evaluation in zip(df["prompt_number"], results):
                if isinstance(evaluation, dict):
                    value = evaluation.get("result", evaluation)
                else:
                    value = evaluation
                writer.writerow([prompt_number, value])


def get_informative_score(IS_score_file):
    scores = []
    df = pd.read_csv(IS_score_file)
    for idx, row in df.iterrows():
        eval_dict = ast.literal_eval(row['evaluation_result'])
        score = eval_dict['score']
        scores.append(score)
    return sum(scores)/len(scores)



