﻿from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk_wiki = "The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania. NLTK includes graphical demonstrations and sample data. It is accompanied by a book that explains the underlying concepts behind the language processing tasks supported by the toolkit, plus a cookbook."

words = word_tokenize(nltk_wiki)

stop_words = stopwords.words("english")

# filtered_sentence = []
# 
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)

filtered_sentence = [w for w in words if w not in stop_words]

print(filtered_sentence)
