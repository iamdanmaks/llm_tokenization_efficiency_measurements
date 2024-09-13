import os
import string
import unicodedata

import pandas as pd

from numpy import median

from tqdm import tqdm

from transformers import AutoTokenizer
from tokenizers.decoders import ByteLevel


def has_cyrillic(character):
    # Get the Unicode name of the character
    char_name = unicodedata.name(character, None)
    if char_name is None:
        return False
    
    if character.lower() in "ёыэъџЊљћјѕќѓў":
        return False

    # Check if 'CYRILLIC' is in the Unicode name
    return character == "▁" or 'CYRILLIC' in char_name


def is_only_symbols(s):
    # Check each character in the string
    for char in s:
        if char.isalnum() or char.isspace():
            return False
    return True


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
    "Xenova/claude-tokenizer",
    "Xenova/gpt-4",
    "Xenova/gpt-4o"
]

tokenizers = [
    AutoTokenizer.from_pretrained(tn) for tn in tokenizers_names
]

decoder = ByteLevel()

eng_letters = string.ascii_lowercase + string.ascii_uppercase + "▁"

stats = {
    "Tokenizer": [],
    "Vocabulary Size": [],
    "English Token Count": [],
    "English Mean CPT": [],
    "English Median CPT": [],
    "English Mean BPT": [],
    "English Median BPT": [],
    "Cyrillic Token Count": [],
    "Cyrillic Mean CPT": [],
    "Cyrillic Median CPT": [],
    "Cyrillic Mean BPT": [],
    "Cyrillic Median BPT": []
}

def gather_stats(tokenizer, tokenizer_name):
    eng_cnt = 0
    eng_token_length = 0
    eng_bytes_token_length = 0
    eng_lengths = []
    eng_lengths_bytes = []

    cyrillic_cnt = 0
    cyrillic_token_length = 0
    cyrillic_bytes_token_length = 0
    cyrillic_lengths = []
    cyrillic_lengths_bytes = []

    for token in tqdm(tokenizer.vocab):
        token = decoder.decode([token])
        
        eng = True
        cyrillic = True
        for char in token:
            if char not in eng_letters:
                eng = False
                break
        for char in token:
            if not has_cyrillic(char):
                cyrillic = False
                break
        if eng:
            eng_cnt += 1
            eng_token_length += len(token)
            eng_bytes_token_length += len(token.encode("utf-8"))
            eng_lengths.append(len(token))
            eng_lengths_bytes.append(len(token.encode("utf-8")))
        elif cyrillic:
            cyrillic_cnt += 1
            cyrillic_token_length += len(token)
            cyrillic_bytes_token_length += len(token.encode("utf-8"))
            cyrillic_lengths.append(len(token))
            cyrillic_lengths_bytes.append(len(token.encode("utf-8")))

    stats["Tokenizer"].append(tokenizer_name)
    stats["Vocabulary Size"].append(len(tokenizer.vocab))
    stats["English Token Count"].append(eng_cnt)
    stats["English Mean CPT"].append(eng_token_length / eng_cnt)
    stats["English Median CPT"].append(median(eng_lengths))
    stats["English Mean BPT"].append(eng_bytes_token_length / eng_cnt)
    stats["English Median BPT"].append(median(eng_lengths_bytes))
    stats["Cyrillic Token Count"].append(cyrillic_cnt)
    stats["Cyrillic Mean CPT"].append(cyrillic_token_length / cyrillic_cnt)
    stats["Cyrillic Median CPT"].append(median(cyrillic_lengths))
    stats["Cyrillic Mean BPT"].append(cyrillic_bytes_token_length / cyrillic_cnt)
    stats["Cyrillic Median BPT"].append(median(cyrillic_lengths_bytes))

for tokenizer_name, tokenizer in zip(tokenizers_names, tokenizers):
    print(f"\n{tokenizer_name}")
    gather_stats(tokenizer, tokenizer_name)

df = pd.DataFrame.from_dict(stats)
df.to_excel("tokenizers_stats.xlsx")
