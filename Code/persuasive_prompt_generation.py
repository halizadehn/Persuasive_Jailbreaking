import os
import re
import pandas as pd
from ollama import chat
from ollama import ChatResponse

def persuasive_prompt_generation():
    with open(os.path.join(prompt_dir, "prompt_rewriting_task.md"), "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()
    df = pd.read_csv(os.path.join(data_dir, "harmful_behaviors.csv"), index_col=None)
    persuasive_prompt_df = pd.DataFrame()
    for i in range(len(df)):
        question = df.loc[i, "goal"]
        full_prompt = f"{base_prompt}\n{question}"
        response: ChatResponse = chat(model='wizardlm-uncensored', messages=[
          {
            'role': 'user',
            'content': full_prompt
          }
        ])
        persuasive_prompt_df.loc[i, "persuasive_prompt"] = response.choices[0].message.content
        persuasive_prompt_df.iloc[[i]].to_csv(
            os.path.join(data_dir, "persuasive_prompts_wizardlm-uncensored_all.csv"),
            index=False, header=False, mode='a'
        )
        print(f"Processed {i+1}/{len(df)}")

