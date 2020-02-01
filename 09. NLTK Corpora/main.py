from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

bible_text = gutenberg.raw('bible-kjv.txt')

tokenized = sent_tokenize(bible_text)

for s in tokenized[0:5]:
    print(s)
