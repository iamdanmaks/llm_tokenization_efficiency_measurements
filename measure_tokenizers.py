import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd

from nltk import sent_tokenize
from nltk import word_tokenize

from nltk.corpus import brown

from nltk.tokenize.treebank import TreebankWordDetokenizer

from tiktoken import get_encoding
from tiktoken import encoding_for_model

from transformers import AutoTokenizer

from transliterate import translit

from tqdm import tqdm

from ua_gec import Corpus


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
    "dataset": [],
    "model": [],
    "bytes_count": [],
    "char_count": [],
    "bytes_per_char": [],
    "word_count": [],
    "char_per_word": [],
    "sentence_count": [],
    "word_per_sentence": [],
    "avg_bytes_per_text": [],
    "avg_chars_per_text": [],
    "avg_words_per_text": [],
    "avg_sentences_per_text": [],
    "token_count": [],
    "avg_token_per_text": [],
    "fertility": []
}


def measure_tokenizer(files, dataset_name, tokenizer, tokenizer_name, from_file=True, transliterate=False):
    print(f"\n{tokenizer_name}")

    is_gpt = tokenizer_name.startswith("gpt")

    bytes_count = 0
    char_count = 0
    word_count = 0
    sentence_count = 0
    token_count = 0

    for file in tqdm(files):
        if from_file:
            with open(file, "r") as f:
                text = f.read()
        else:
            text = file
        
        if transliterate:
            text = translit(text, "uk", reversed=True)
        
        bytes_count += len(text.encode('utf-8'))
        char_count += len(text)
        word_count += len(word_tokenize(text))
        sentence_count += len(sent_tokenize(text))
        
        if not is_gpt:
            token_count += len(tokenizer.tokenize(text))
        else:
            token_count += len(tokenizer.encode(text))

    results["dataset"].append(dataset_name)
    results["model"].append(tokenizer_name)
    results["bytes_count"].append(bytes_count)
    results["char_count"].append(char_count)
    results["bytes_per_char"].append(bytes_count / char_count)
    results["word_count"].append(word_count)
    results["char_per_word"].append(char_count / word_count)
    results["sentence_count"].append(sentence_count)
    results["word_per_sentence"].append(word_count / sentence_count)
    results["avg_bytes_per_text"].append(bytes_count / len(files))
    results["avg_chars_per_text"].append(char_count / len(files))
    results["avg_words_per_text"].append(word_count / len(files))
    results["avg_sentences_per_text"].append(sentence_count / len(files))
    results["token_count"].append(token_count)
    results["avg_token_per_text"].append(token_count / len(files))
    results["fertility"].append(token_count / word_count)

print("\n\n\n===== Ukrainian Brown Corpus =====")
ukr_brown_good = [f"./brown_corpus/good/{fn}" for fn in os.listdir("./brown_corpus/good")]
ukr_brown_bad = [f"./brown_corpus/bad/{fn}" for fn in os.listdir("./brown_corpus/bad")]
ukr_brown_so = [f"./brown_corpus/so-so/{fn}" for fn in os.listdir("./brown_corpus/so-so")]
ukr_brown = ukr_brown_good + ukr_brown_bad + ukr_brown_so

for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(ukr_brown, "Ukrainian Brown Corpus", tokenizer, tokenizers_names[i])

print("\n\n\n===== Ukrainian Transliterated Brown Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(ukr_brown, "Ukrainian Transliterated Brown Corpus", tokenizer, tokenizers_names[i], transliterate=True)

print("\n\n\n===== English Brown Corpus =====")

eng_brown = []
for cat in brown.categories():
    eng_brown.append(TreebankWordDetokenizer().detokenize(brown.words(categories=cat)))

for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(eng_brown, "English Brown Corpus", tokenizer, tokenizers_names[i], from_file=False)

law_df = pd.read_csv("./domains/train_laws.csv", sep="|")
texts = law_df["fixed_uk"].tolist()

print("\n\n\n===== Ukrainian Laws Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(texts, "Ukrainian Laws Corpus", tokenizer, tokenizers_names[i], from_file=False)

texts = law_df["en"].tolist()

print("\n\n\n===== English Translation of Ukrainian Laws Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(texts, "English Translation of Ukrainian Laws Corpus", tokenizer, tokenizers_names[i], from_file=False)

science_df = pd.read_csv("./domains/train_science.csv", sep="|")
texts = science_df["fixed_uk"].tolist()

print("\n\n\n===== Ukrainian Science Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(texts, "Ukrainian Science Corpus", tokenizer, tokenizers_names[i], from_file=False)

texts = science_df["en"].tolist()

print("\n\n\n===== English Translation of Ukrainian Science Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(texts, "English Translation of Ukrainian Science Corpus", tokenizer, tokenizers_names[i], from_file=False)

corpus = Corpus(partition="train", annotation_layer="gec-only")
texts = [doc.source for doc in corpus if len(word_tokenize(doc.source)) == len(word_tokenize(doc.target))]

print("\n\n\n===== Ukrainian GEC Incorrect Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(texts, "Ukrainian GEC Incorrect Corpus", tokenizer, tokenizers_names[i], from_file=False)

corpus = Corpus(partition="train", annotation_layer="gec-only")
texts = [doc.target for doc in corpus if len(word_tokenize(doc.source)) == len(word_tokenize(doc.target))]

print("\n\n\n===== Ukrainian GEC Correct Corpus =====")
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(texts, "Ukrainian GEC Correct Corpus", tokenizer, tokenizers_names[i], from_file=False)

print("\n\n\n===== Ukrainian Code Documentation Corpus =====")
ukr_vue_docs = [f"./domains/docs/uk/{fn}" for fn in os.listdir("./domains/docs/uk")]
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(ukr_vue_docs, "Ukrainian Code Documentation Corpus", tokenizer, tokenizers_names[i])

print("\n\n\n===== English Code Documentation Corpus =====")
ukr_vue_docs = [f"./domains/docs/en/{fn}" for fn in os.listdir("./domains/docs/en")]
for i, tokenizer in enumerate(tokenizers):
    measure_tokenizer(ukr_vue_docs, "English Code Documentation Corpus", tokenizer, tokenizers_names[i])

df = pd.DataFrame.from_dict(results)
df.to_excel("tokenizers_measurements.xlsx")
