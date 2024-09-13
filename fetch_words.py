import pandas as pd

from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm


BASE_URL = "https://slovnyk.ua/"

all_words = []

for i in range(1, 33):
    try:
        print(f"\n\n\nLetter number #{i}")
        html_content = get(f"https://slovnyk.ua/index.php?s1={i}&s2=0").text

        soup = BeautifulSoup(html_content, 'html.parser')
        word_group_links = [a['href'] for a in soup.find_all('a', class_='cont_link')]

        for ind, word_group_link in enumerate(word_group_links[:10]):
            print(f"Group {ind + 1} / {len(word_group_links)}")
            html_content = get(BASE_URL + word_group_link).text
            soup = BeautifulSoup(html_content, 'html.parser')
            words = [a.get_text(strip=True) for a in soup.find_all('a', class_='cont_link') if len(a.get_text(strip=True)) > 2]
            all_words += words
    except KeyboardInterrupt:
        pass

df = pd.DataFrame.from_dict({
    "words": all_words
})
df.to_excel("words.xlsx")
