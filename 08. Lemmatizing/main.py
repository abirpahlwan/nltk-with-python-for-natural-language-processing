from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def display(words):
    for w in words:
        print(lemmatizer.lemmatize(w))


test_words = ['cats', 'geese', 'cacti', 'corpora']

display(test_words)

print(lemmatizer.lemmatize('better', pos='a'))
