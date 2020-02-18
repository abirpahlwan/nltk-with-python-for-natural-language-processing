from nltk.corpus import wordnet

# synset = wordnet.synsets('program')

# print(synset)
# print(synset[0].lemmas())
# print(synset[0].lemmas()[0].name())
# print(synset[0].definition())
# print(synset[1].examples())

'''for syn in wordnet.synsets('good'):
    for lemma in syn.lemmas():
        print(lemma.name())
        if lemma.antonyms():
            for antonym in lemma.antonyms():
                print(antonym.name())
        print('--------------------')'''


def check_similarity(first_word, second_word):
    print(first_word, second_word, wordnet.synsets(first_word)[0].wup_similarity(wordnet.synsets(second_word)[0]))


check_similarity('car', 'ship')
check_similarity('car', 'cat')
check_similarity('car', 'phone')
check_similarity('car', 'wheel')
