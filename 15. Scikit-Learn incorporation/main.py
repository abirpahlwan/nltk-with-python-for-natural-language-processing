import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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
    features = {}    # Empty dictionary named 'features'
    for word in word_features:
        features[word] = (word in words)    # Check if 'word' exists in the 'words' set

    return features


feature_sets = [(find_features(review), category) for (review, category) in documents]

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]


"""Naive Bayes Classifier"""
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Classifier accuracy: ",
      (nltk.classify.accuracy(classifier, testing_set))*100)


"""Multinomial Naive Bayes Classifier"""
multinomial_naive_bayes_classifier = SklearnClassifier(MultinomialNB())
multinomial_naive_bayes_classifier.train(training_set)
print("Multinomial Naive Bayes Classifier accuracy: ",
      (nltk.classify.accuracy(multinomial_naive_bayes_classifier, testing_set))*100)


"""Gaussian Naive Bayes Classifier"""
# gaussian_naive_bayes_classifier = SklearnClassifier(GaussianNB())
# gaussian_naive_bayes_classifier.train(training_set)
# print("Gaussian Naive Bayes Classifier accuracy: ",
#       (nltk.classify.accuracy(gaussian_naive_bayes_classifier, testing_set))*100)


"""Bernoulli Naive Bayes Classifier"""
bernoulli_naive_bayes_classifier = SklearnClassifier(BernoulliNB())
bernoulli_naive_bayes_classifier.train(training_set)
print("Bernoulli Naive Bayes Classifier accuracy: ",
      (nltk.classify.accuracy(bernoulli_naive_bayes_classifier, testing_set))*100)


"""Logistic Regression Classifier"""
logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
print("Logistic Regression Classifier accuracy: ",
      (nltk.classify.accuracy(logistic_regression_classifier, testing_set))*100)


"""Stochastic Gradient Descent Classifier"""
stochastic_gradient_descent_classifier = SklearnClassifier(SGDClassifier())
stochastic_gradient_descent_classifier.train(training_set)
print("Stochastic Gradient Descent Classifier accuracy: ",
      (nltk.classify.accuracy(stochastic_gradient_descent_classifier, testing_set))*100)


"""Support Vector Classifier"""
support_vector_classifier = SklearnClassifier(SVC())
support_vector_classifier.train(training_set)
print("Support Vector Classifier accuracy: ",
      (nltk.classify.accuracy(support_vector_classifier, testing_set))*100)


"""Linear Support Vector Classifier"""
linear_support_vector_classifier = SklearnClassifier(LinearSVC())
linear_support_vector_classifier.train(training_set)
print("Linear Support Vector Classifier accuracy: ",
      (nltk.classify.accuracy(linear_support_vector_classifier, testing_set))*100)


"""Nu Support Vector Classifier"""
n_support_vector_classifier = SklearnClassifier(NuSVC())
n_support_vector_classifier.train(training_set)
print("Nu Support Vector Classifier accuracy: ",
      (nltk.classify.accuracy(n_support_vector_classifier, testing_set))*100)
