import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd

from tiktoken import get_encoding
from tiktoken import encoding_for_model

from transformers import AutoTokenizer

from tqdm import tqdm


def get_openai_tokenizer(tokenizer_name):
    if tokenizer_name == "gpt-4o":
        get_encoding("o200k_base")
    else:
        get_encoding("cl100k_base")

    return encoding_for_model(tokenizer_name)


tokenizers_names = [
    "openai-community/gpt2",
    "malteos/gpt2-uk",
    "meta-llama/Llama-2-7b-chat-hf",
    "mattshumer/Reflection-Llama-3.1-70B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-7b-it",
    "Qwen/Qwen2-VL-7B-Instruct",
    "microsoft/phi-2",
    "microsoft/Phi-3.5-MoE-instruct",
    "state-spaces/mamba-2.8b-hf",
    "Xenova/claude-tokenizer"
]

tokenizers = [
    AutoTokenizer.from_pretrained(tn) for tn in tokenizers_names
]

openai_tokenizers = [
    "gpt-4",
    "gpt-4o"
]

tokenizers_names += openai_tokenizers

tokenizers += [
    get_openai_tokenizer(tn) for tn in openai_tokenizers
]

results = {
    "Model": [],
    "Nominative": [],
    "Genitive": [],
    "Dative": [],
    "Accusative": [],
    "Instrumental": [],
    "Locative": [],
    "Vocative": []
}


def measure(words, grammar_case, tokenizer_name, tokenizer):
    is_gpt = tokenizer_name.startswith("gpt")
    token_count = 0

    for word in words:
        if not is_gpt:
            token_count += len(tokenizer.tokenize(word))
        else:
            token_count += len(tokenizer.encode(word))
    
    results[grammar_case].append(token_count / len(words))


df = pd.read_excel("word_forms.xlsx")

for i, tokenizer in enumerate(tokenizers):
    print(f"\n{tokenizers_names[i]}")
    results["Model"].append(tokenizers_names[i])

    for c in tqdm(df.columns[1:]):
        measure(df[c], c, tokenizers_names[i], tokenizer)

results_df = pd.DataFrame.from_dict(results)
results_df.to_excel("tokenizers_grammar_cases_measurements.xlsx")
