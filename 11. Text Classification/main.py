import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
# print(documents[1])

all_words = []
for word in movie_reviews.words():
    all_words.append(word.lower())

word_frequency = nltk.FreqDist(all_words)
# print(word_frequency.most_common(20))
print('fuck', word_frequency['fuck'])
print('shit', word_frequency['shit'])
print('stupid', word_frequency['stupid'])
