import os
import numpy as np
import nltk
import re
from collections import OrderedDict


class LanguageModel():

    def __init__(self, sentences, n):
        self.sentences = sentences
        self.n = n
        self.n_grams = {}
        self.char_grams = {}
        self.probabilities = {}
        self.wb_probabilities = {}
        self.char_probabilities = {}

    def generate_model(self):
        for counter, value in enumerate(self.sentences):
            value = re.findall(
                "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", value)
            self.n_grams = self.get_ngram(value)

    def get_ngram(self, value, char=False):
        temp = {}
        if char == True:
            #             print "Hey"
            temp = self.char_grams
        else:
            temp = self.n_grams
        if self.n == 1:
            l = len(value)
            for index in range(0, l):
                unit = value[index]
                if unit not in temp:
                    temp[unit] = 0
                temp[unit] += 1
        else:
            l = len(value) - self.n + 1
            for index in range(0, l):
                unit = tuple(value[index:index + self.n])
                if unit not in temp:
                    temp[unit] = 0
                temp[unit] += 1
        return temp

    def char_ngram(self, tokens):
        # For charachter n_grams neeeded for spell correction
        for token in tokens:
            letters = list(token)
            self.char_grams = self.get_ngram(letters, True)

    def sort_ngrams(self, char=False):
        if char:
            return sorted(self.char_grams.items(), key=lambda x: x[1], reverse=True)
        return sorted(self.n_grams.items(), key=lambda x: x[1], reverse=True)

    def get_model(self):
        return self.n_grams

    def get_char_grams(self):
        return self.char_grams

    def get_probabilities(self):
        return self.probabilities

    def get_wb_probabilities(self):
        return self.wb_probabilities

    def get_char_probabilities(self):
        return self.char_probabilities

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    def set_wb_probabilities(self, probabilities):
        self.wb_probabilities = probabilities

    def set_char_probabilities(self, probabilities):
        self.char_probabilities = probabilities


def train():
    classes = ['ArtOrDet', 'Cit', 'Mec', 'Nn', 'Npos', 'Others', 'Pform', 'Pref', 'Prep', 'Rloc-',
               'SVA', 'Sfrag', 'Smod', 'Spar', 'Srun', 'Ssub', 'Trans', 'Um', 'V0', 'Vform', 'Vm',
               'Vt', 'WOadv', 'WOinc', 'Wa', 'Wci', 'Wform', 'Wtone']

    # files = os.listdir(os.getcwd() + '/Data')
    train_data = open(os.getcwd() + '/Data/train.txt', "r")
    # with open(os.getcwd() + '/Data/train.txt', "r") as myfile:
    data = train_data.read()
    sentences = nltk.sent_tokenize(data)
    sentence_dict = []
    for counter, sentence in enumerate(sentences):
        lines = sentence.split("\n")
        dic = OrderedDict()
        for line in lines:
            words = line.split()
            if not words:
                continue
            if (len(words) > 1 and words[-1] in classes):
                dic[words[0]] = (words[1:-1], words[-1])
            else:
                dic[words[0]] = (None, None)
        sentence_dict.append(dic)
    print sentence_dict[0]
    return sentence_dict, classes

    train_data.close()
    # train_data = open(os.getcwd() + '/Data/train.txt', "r")

    # correction = []
    # error_category = []
    # words = []
    # word_class = []
    # # print "HEy"
    # for counter, line in enumerate(train_data):
    #     # print counter
    #     line_list = line.split()
    #     # print "HeY"
    #     # print counter
    #     # print line_list
    #     if not line_list:
    #         continue
    #     # if line_list[-1] == ")":
    #     #     print line_list
    #     # print line_list
    #     words.append(line_list[0])
    #     if(len(line_list) > 1):
    #         correction.append(line_list[1:-1])
    #         error_category.append(line_list[-1])

    # correction = np.asarray(correction)
    # error_category_array = np.asarray(error_category)
    # classes = np.unique(error_category_array)
    # words_joined = " ".join(words)
    # # regex = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    # # words_joined = re.split(words_joined)
    # words_joined = nltk.sent_tokenize(words_joined)
    # return correction, words_joined, classes


def make_ngrams(keyList, counter, key):
    l = len(keyList)
    unigrams = []
    bigrams = []
    trigrams = []
    if counter == 0:
        trigrams.append(tuple(keyList[0:3]))
        bigrams.append(tuple(keyList[0:2]))
        unigrams.append(tuple(keyList[counter]))
    elif counter == 1:
        trigrams.append(tuple(keyList[0:3]))
        trigrams.append(tuple(keyList[1:4]))
        bigrams.append(tuple(keyList[counter: counter + 2]))
        bigrams.append(tuple(keyList[counter - 1:counter + 1]))
        unigrams.append(tuple(keyList[counter]))
        # trigrams.append(tuple(keyList[counter - 2:counter + 1]))
    elif counter == l - 1:
        trigrams.append(tuple(keyList[counter - 2:l]))
        bigrams.append(tuple(keyList[counter - 1:l]))
        unigrams.append(tuple(keyList[counter]))

    elif counter == l - 2:
        trigrams.append(tuple(keyList[counter - 2:l - 1]))
        trigrams.append(tuple(keyList[counter - 1:l]))
        bigrams.append(tuple(keyList[counter: counter + 2]))
        bigrams.append(tuple(keyList[counter - 1:counter + 1]))
        unigrams.append(tuple(keyList[counter]))

    else:
        trigrams.append(tuple(keyList[counter:counter + 3]))
        trigrams.append(tuple(keyList[counter - 1:counter + 2]))
        trigrams.append(tuple(keyList[counter - 2:counter + 1]))
        bigrams.append(tuple(keyList[counter: counter + 2]))
        bigrams.append(tuple(keyList[counter - 1:counter + 1]))
        unigrams.append(tuple(keyList[counter]))

    return unigrams, bigrams, trigrams


def naive_bayes(error_category):
    pass

if __name__ == "__main__":
    sentence_dict, classes = train()
    class_unigrams = {}
    class_bigrams = {}
    class_trigrams = {}
    for counter, sentence in enumerate(sentence_dict):
        sentence_joined = " ".join(sentence.keys())
        keyList = (sentence.keys())
        for counter, key in enumerate(keyList):
            if sentence[key][1] is not None:
                unigrams, bigrams, trigrams = make_ngrams(
                    keyList, counter, key)
                for unigram in unigrams:
                    if sentence[key][1] not in class_unigrams:
                        class_unigrams[sentence[key][1]] = {}
                    if unigram not in class_unigrams[sentence[key][1]]:
                        class_unigrams[sentence[key][1]][unigram] = 0
                class_unigrams[sentence[key][1]][unigram] += 1

                for bigram in bigrams:
                    if sentence[key][1] not in class_bigrams:
                        class_bigrams[sentence[key][1]] = {}
                    if bigram not in class_bigrams[sentence[key][1]]:
                        class_bigrams[sentence[key][1]][bigram] = 0
                class_bigrams[sentence[key][1]][bigram] += 1

                for trigram in trigrams:
                    if sentence[key][1] not in class_trigrams:
                        class_trigrams[sentence[key][1]] = {}
                    if trigram not in class_trigrams[sentence[key][1]]:
                        class_trigrams[sentence[key][1]][trigram] = 0
                class_trigrams[sentence[key][1]][trigram] += 1

        print class_trigrams.values()
        # for key, value in sentence.iteritems():
        #     if value[1] != None:

                # print value[1]

                # print sentence_joined

                # print sentences[3]
