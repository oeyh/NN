import numpy as np
import re
from nltk.stem import PorterStemmer
from sklearn.svm import SVC
from scipy.io import loadmat


def getVocabList():
    """ reads the vocabulary list in vocab.txt
        and returns a dictionary of the words in vocabList.
    """

    d = {}

    with open('vocab.txt', 'r') as f:
        for line in f:
            li = line.split()
            d[li[1]] = int(li[0])

    return d


def getVocabList2():
    """ reads the vocabulary list in vocab.txt
        and returns a dictionary of the words in vocabList.
        In this dictionary, integer index is key, word string is value
    """

    d = {}

    with open('vocab.txt', 'r') as f:
        for line in f:
            li = line.split()
            d[int(li[0])] = li[1]

    return d


def processEmail(email_contents):
    """ preprocesses the body of an email and returns a list of indices of the
        words contained in the email.
    """

    # init
    result_indices = []
    result_words = []  # for debugging
    vocabDict = getVocabList()

    # prepocessing
    # all lowercase
    email_contents = email_contents.lower()

    # strip HTML tags
    # starts with <, ends with >, has 0 or 1 /, no >, 1 or more chars
    p = re.compile(r'\<[^<>]+\>')
    email_contents = p.sub(' ', email_contents)

    # replace numbers with 'number'
    p = re.compile(r'[0-9]+')
    email_contents = p.sub('number', email_contents)

    # replace urls with 'httpaddr'
    p = re.compile(r'http(s?)://(\S*)')
    email_contents = p.sub('httpaddr', email_contents)

    # replace email address with 'emailaddr'
    p = re.compile(r'\S+@\S+')
    email_contents = p.sub('emailaddr', email_contents)

    # replace dollar sign $ with 'dollar'
    p = re.compile(r'\$+')
    email_contents = p.sub('dollar', email_contents)

    # tokenize and get rid of any punctuation and non alphanumerics
    content_list = re.split(r'[\W_]', email_contents)

    # debugging
    print(' '.join(content_list))

    # word stemming
    stemmer = PorterStemmer(mode='ORIGINAL_ALGORITHM')
    for word in content_list:
        # only stem those words with len > 2
        # leave as is for len = 1 or 2
        if len(word) > 2:
            word = stemmer.stem(word)
            if word in vocabDict:
                result_indices.append(vocabDict[word])
                result_words.append(word)
        elif len(word) >= 1:
            if word in vocabDict:
                result_indices.append(vocabDict[word])
                result_words.append(word)

    ### debugging ###
    #print(' '.join(content_list))

    return result_indices, result_words


def comparePM(txtfile, matfile):
    """ compare results in Python and Matlab. part of debugging process.
    """

    # Python calculation
    with open(txtfile, 'r') as f:
        file_contents = f.read()
    word_indices, word_list = processEmail(file_contents)

    # MATLAB result
    data = loadmat(matfile)
    matlab_indices = data['word_indices'].ravel().tolist()

    # vocabulary dict
    vocabDict = getVocabList()
    vocabDict2 = getVocabList2()

    matlab_words = [vocabDict2[i] for i in matlab_indices]

    # Compare length of list
    print("### Comparing length of list ###")
    print("Matlab: {}".format(len(matlab_indices)))
    print("Python: {}".format(len(word_indices)))
    print()

    # Compare word indices
    print("### Comparing word indices ###")
    print("Matlab: ")
    print(matlab_indices)
    print()
    print("Python: ")
    print(word_indices)
    print()

    # Compare preprocessed email contents
    print("### Comparing preprocessing results ###")
    print("Matlab: ")
    print(' '.join(matlab_words))
    print()
    print("Python: ")
    print(' '.join(word_list))


def main():

    txtfile = 'emailSample2.txt'
    matfile = 'emailSample2.mat'
    comparePM(txtfile, matfile)


if __name__ == '__main__':
    main()
