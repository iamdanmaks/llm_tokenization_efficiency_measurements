import pymorphy2

import pandas as pd

from tqdm import tqdm


morph = pymorphy2.MorphAnalyzer(lang='uk')
df = pd.read_excel("words.xlsx")

result = {
    "Nominative": [],
    "Genitive": [],
    "Dative": [],
    "Accusative": [],
    "Instrumental": [],
    "Locative": [],
    "Vocative": []
}

case2pm = {
    "Nominative": "nomn",
    "Genitive": "gent",
    "Dative": "datv",
    "Accusative": "ablt",
    "Instrumental": "accs",
    "Locative": "loct",
    "Vocative": "voct",
}

for word in tqdm(df["words"]):
    missing = False

    temp = {
        "nomn": "",
        "gent": "",
        "datv": "",
        "ablt": "",
        "accs": "",
        "loct": "",
        "voct": ""
    }

    for case, pm_case in case2pm.items():
        word_form = morph.parse(word)[0].inflect({pm_case})
        if not word_form:
            missing = True
            break
        temp[case] = word_form.word
    
    if not missing:
        for case in case2pm:
            result[case].append(temp[case])

print("words count", len(result["Nominative"]))

result_df = pd.DataFrame.from_dict(result)
result_df.to_excel("word_forms.xlsx")
