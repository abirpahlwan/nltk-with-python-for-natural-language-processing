import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for word in movie_reviews.words():
    all_words.append(word.lower())

word_frequency = nltk.FreqDist(all_words)

word_features = list(word_frequency.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}  # Empty dictionary named 'features'
    for word in word_features:
        features[word] = (word in words)    # Check if 'word' exists in the 'words' set

    return features


# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

feature_sets = [(find_features(review), category) for (review, category) in documents]

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Classifier accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
