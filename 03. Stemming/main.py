from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()


def display(words):
    for w in words:
        print(ps.stem(w))


# test_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']
# display(test_words)

test_sentence = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

test_words = word_tokenize(test_sentence)
display(test_words)
