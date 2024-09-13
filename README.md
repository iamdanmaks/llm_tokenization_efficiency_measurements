# LLM tokenization efficiency measurements

This repository contains a set of scripts to measure the efficiency of LLMs tokenizers for Ukrainian language (both in terms of general texts and narrow domains ones).

Measurements are conducted on:

- Ukrainian Brown corpus
- English Brown corpus
- Laws translations scrapped from the Verkhovna Rada website
- Scientific articles abstracts translations
- VueJS documentation translations
- Ukrainian GEC dataset (only those pairs that have the same number of words for both correct and incorrect versions)

  You can get a total token count, average tokens per text, and tokenization fertility with provided scripts for a set of models from huggingface hub and tiktoken.

  - measure_tokenizers.py - provides fertility measurements for all listed datasets
  - measure_abc.py - provides measurements for the Ukrainian alphabet to check if the model can tokenize each letter individually or would fall back to bytes
  - measure_grammar_cases.py - provides measurements for all 7 Ukrainian grammar cases (відмінки) to check the influence of word form changing
  - tokenizer_stats.py - measure the vocabulary size of tokenizers and count English and Cyrillic tokens in their vocabularies
  - fetch_words.py - fetches words from the Ukrainian online dictionary to generate multiple word forms later with PyMorphy2
  - get_grammer_cases.py - generates word forms for each grammar case for all obtained words. If it is impossible to generate a grammar case for the word, it will be skipped

Scripts require a `HF_TOKEN` environment variable to access models repositories on huggingface hub.

For Ukrainian Brown corpus download folders `good`, `bad`, `so=so` from here: https://github.com/brown-uk/corpus/tree/master/data. Put them in `brown_corpus` subfolder in this repository to run measurements.
