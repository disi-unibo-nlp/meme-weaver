import re
import json
import nltk
from collections import Counter, defaultdict

stopword_en = nltk.corpus.stopwords.words('english')
stopword_es = nltk.corpus.stopwords.words('spanish')
stopwords = stopword_en + stopword_es


def flatten(xss):
    return [x for xs in xss for x in xs]


def tokenize(text):
    """
    Tokenizes text into words/numbers, converts to lower-case, and removes tokens that:
      - are in the stopwords set
      - are 2 or fewer characters long.
    """
    # Extract tokens (words and numbers)
    tokens = re.findall(r'\w+[\+\w]*', text.lower())
    # Filter out stopwords and tokens with 2 or fewer characters.
    tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    return tokens


def get_bigrams(tokens):
    return [ (tokens[i], tokens[i+1]) for i in range(len(tokens)-1) ]

path = "exist2024_memes_dataset/training/EXIST2024_training.json"


# Open and read the JSON file
with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

task6_labels = []
meme_texts = []
for meme_id, meme_data in data.items():
    meme_texts.append(meme_data['text'])
    task6_labels.append(meme_data['labels_task6'])



# Create a dictionary that will hold counts per label:
# { label_name: { 'unigrams': Counter(...), 'bigrams': Counter(...) } }
label_counts = defaultdict(lambda: {'unigrams': Counter(), 'bigrams': Counter()})


# IMPORTANT: This code assumes that each text is segmented into as many segments as there are label groups.
# Here, we split the text into roughly equal parts (by word count).
for text, labels in zip(meme_texts, task6_labels):
    unigrams = tokenize(text)
    flatten_labels = flatten(labels)

    for label in flatten_labels:
        bigrams = get_bigrams(unigrams)
        for unigr in unigrams:
            label_counts[label]['unigrams'].update([unigr])

        for bigr in bigrams:
            label_counts[label]['bigrams'].update([bigr])


ordered_bigrams = {}
ordered_unigrams = {}
for label, counts in label_counts.items():
    ordered_unigrams[label] = counts['unigrams'].most_common()
    ordered_bigrams[label] = counts['bigrams'].most_common()

