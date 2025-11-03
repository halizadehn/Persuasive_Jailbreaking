# Persuasion-Aware Jailbreaking as Social Engineering: Fingerprinting LLM Susceptibilities and Defense Implications

This repository contains the implementation and evaluation framework for studying persuasion-based jailbreak attacks on Large Language Models (LLMs). The project operationalizes Cialdini’s persuasion principles to generate adversarial prompts and analyze model-specific susceptibility profiles.

## Pipeline overview
![overview](./Figures/framework.jpg)


## Datasets

- **AdvBench**: This dataset contains 520 queries, covering various types of harmful behavior. 
- **StrongREJECT**: This dataset encompasses six major categories of behaviors that are explicitly prohibited under all major LLM usage policies:
  (1) illegal goods and services, (2) non-violent crimes, (3) hate, harassment, and discrimination, (4) disinformation and deception, (5) violence, and (6) sexual content. 

## Victim Models

- Vicuna-7b
- Llama2-7b-chat
- Llama3
- DeepSeek-R1
- Gemma3
- Phi4

> **Note:** Target models are available via [Ollama](https://ollama.com/).

## Attack Baselines
- **[GCG](https://github.com/llm-attacks/llm-attacks)**: finds suffixes via gradient synthesis
- **[PAIR](https://arxiv.org/pdf/2310.08419)**: an algorithm that creates semantic jailbreaks using only black-box access to an LLM
- **[PAP](https://github.com/CHATS-lab/persuasive_jailbreaker)**: generates persuasive adversarial prompts using different strategies

  > **Note:** We leveraged baseline implementations provided by the [StrongReject benchmark](https://github.com/dsbowen/strong_reject).

## Code Structure

- [persuasive_prompt_generation.py](./Code/persuasive_prompt_generation.py): Generates persuasive variants of harmful queries based on Cialdini’s persuasion principles. These variants are used to analyze how different persuasion strategies influence LLM compliance with harmful instructions.

- [model_response_collector.py](./Code/model_response_collector.py): Handles the process of querying LLMs with both original and persuasion-based prompts, collecting model outputs to measure refusal behavior and susceptibility to persuasion-driven jailbreaks.

- [evaluation.py](./Code/evaluation.py): Evaluates collected responses using metrics such as Attack Success Rate (ASR), Informativeness Score (IS), and Influence Power (IP) to quantify the effectiveness and impact of various persuasion strategies across models.

- [Attack baselines](./Code/Attack baselines/): Contains baseline implementations of traditional adversarial prompt generation techniques, serving as benchmarks to compare against persuasion-aware jailbreak methods.
  
- [Defense baselines](./Code/Defense baselines/): Includes implementations of defense strategies used to evaluate the robustness of LLMs against persuasion-based attacks.


