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
    "meta-llama/Meta-Llama-3.1-8B",
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

abc = "АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщьЮюЯя'"
abc = [_ for _ in abc]

results = {
    "Model": [],
    "Number of known letters": [],
    "Number of unknown letters": [],
    "Fertility": []
}

def measure(letters, tokenizer_name, tokenizer):
    is_gpt = tokenizer_name.startswith("gpt")
    token_count = 0
    known = 0
    unknown = 0

    for letter in tqdm(letters):
        if not is_gpt:
            tokenization = len(tokenizer.tokenize(letter))
        else:
            tokenization = len(tokenizer.encode(letter))
        
        if tokenization > 1:
            unknown += 1
        else:
            known += 1
        
        token_count += tokenization
    
    results["Number of known letters"].append(known)
    results["Number of unknown letters"].append(unknown)
    results["Fertility"].append(token_count / len(letters))


df = pd.read_excel("word_forms.xlsx")

for i, tokenizer in enumerate(tokenizers):
    print(f"\n{tokenizers_names[i]}")
    results["Model"].append(tokenizers_names[i])

    measure(abc, tokenizers_names[i], tokenizer)

result_df = pd.DataFrame.from_dict(results)
result_df.to_excel("tokenizers_abc_measurements.xlsx")
