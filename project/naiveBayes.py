import math
import os
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.io
import nltk.data
import nltk.tokenize
import nltk.stem
from nltk.corpus import stopwords
from collections import Counter

from IPython.display import display

SMOOTHNING_FACTOR = 1
world_label_matrix = []
NUM_OF_LABELS = 2

label_index = {
    "FAKE": 0,
    "REAL": 1
}

def extract_words(text, stemmer = None, remove_stopwords = False):
    """
    Strategy used:
    1. Tokenize
    2. Stemming
    3. Stop word removal
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    if stemmer is None:
        words = [token.lower() for token in tokens]
    else:
        words = [stemmer.stem(word.lower()) for word in tokens]
    if remove_stopwords:
        words = [word for word in words if word not in stopwords.words('english')]

    return words

def build_vocabulary(documents):
    """
    Creating a list of the total unique worlds found in the corpus.
    """
    vocabulary = set()
    for doc in documents:
        vocabulary.update([word for word in doc])
    vocabulary = list(vocabulary)
    return vocabulary

def get_word_count_by_label(documents):
    """
    Returns the total number of words in each class including
    muliple occurences of the same world
    """
    label_word_count = {
        "FAKE": 0,
        "REAL": 0
    }
    for index, row in documents.iterrows():
        # to skip the first line which is of not importance to us
        if row["label"].strip().upper() == "LABEL":
            continue
        else:
            label_word_count[row["label"].strip().upper()] += len(row["Transformed text"])
    return label_word_count

def calculate_prior_probabilities(documents):
    """
    Returns the Prior probabilies of FAKE/REAL classes from the corpus
    """
    label_priors = {
        "FAKE": 0,
        "REAL": 0
    }
    for index, row in documents.iterrows():
        # to skip the first line which is of not importance to us
        if row["label"].strip().upper() == "LABEL":
            continue
        else:
            label_priors[row["label"].strip().upper()] += 1
    return label_priors

def init_label_word_count_matrix(num_of_words):
    matrix = []
    for i in range(num_of_words):
        matrix.append([0 for n in range(NUM_OF_LABELS)])
    return matrix

def label_word_count_matrix(vocabulary, documents):
    num_of_words = len(vocabulary)
    matrix = init_label_word_count_matrix(num_of_words)
    for row_index in range(num_of_words):
        for index, row in documents.iterrows():
            cleaned_label = row["label"].strip().upper()
            if cleaned_label == "LABEL":
                continue
            col_index = label_index[cleaned_label]
            matrix[row_index][col_index] += row["Transformed text"].count(vocabulary[row_index])
    return matrix

def prob_of_wrd_gvn_class(vocabulary, fake_word_count, real_word_count, label_word_matrix, word):
    """
    Computes the probabilities for the given text for given class
    """
    num_of_words = len(vocabulary)
    try:
        word_index = vocabulary.index(word)
        word_frequency_in_fake_class = label_word_matrix[word_index][label_index["FAKE"]]
        word_frequency_in_real_class = label_word_matrix[word_index][label_index["REAL"]]
    except ValueError as err:
        # word is not present in the vocabulary contructed
        word_frequency_in_fake_class = word_frequency_in_real_class = 0
    prob_of_wrd_gvn_fake = float(
        word_frequency_in_fake_class + SMOOTHNING_FACTOR)/float(
        fake_word_count + num_of_words)
    prob_of_wrd_gvn_real = float(
        word_frequency_in_real_class + SMOOTHNING_FACTOR)/float(
        real_word_count + num_of_words)
    return prob_of_wrd_gvn_fake, prob_of_wrd_gvn_real


def multinomial_NBC(
    new_sample, vocabulary, fake_wrd_cnt, real_wrd_cnt, label_wrd_matrix, prior_of_fake, prior_of_real):
    words = extract_words(new_sample)
    product_gvn_fake = prior_of_fake
    product_gvn_real = prior_of_real
    for word in words:
        prob_of_wrd_gvn_fake, prob_of_wrd_gvn_real = prob_of_wrd_gvn_class(
                vocabulary,
                fake_wrd_cnt,
                real_wrd_cnt,
                label_wrd_matrix, word)
        product_gvn_fake*=prob_of_wrd_gvn_fake
        product_gvn_real*=prob_of_wrd_gvn_real

    print "Probability of word|fake", product_gvn_fake
    print "Probability of word|real", product_gvn_real
    if product_gvn_fake > product_gvn_real:
        return "FAKE"
    else:
        return "REAL"

def main():
    try:
        dataDF = pd.read_csv(
            "small_dataset.csv",
            sep=',', lineterminator='\n',
            names = ["title", "text", "label"], encoding="utf-8")
        snowball = nltk.stem.snowball.EnglishStemmer()
        dataDF["Transformed text"] = dataDF.apply(
            lambda row: extract_words(row['text'], snowball, True), axis=1)
        display(dataDF)
        # Vocabulary building happens with the transformed text
        vocabulary = build_vocabulary(dataDF["Transformed text"])
        num_of_wrds = len(vocabulary)
        print "|VOCABULARY|", num_of_wrds
        label_word_count = get_word_count_by_label(dataDF)
        label_priors = calculate_prior_probabilities(dataDF)
        matrix = label_word_count_matrix(vocabulary, dataDF)
        prior_of_real = float(label_priors["REAL"])/float(label_priors["REAL"] + label_priors["FAKE"])
        prior_of_fake = float(label_priors["FAKE"])/float(label_priors["REAL"] + label_priors["FAKE"])
        real_wrd_cnt = label_word_count["REAL"]
        fake_wrd_cnt = label_word_count["FAKE"]
        print "|W(REAL)|", real_wrd_cnt
        print "|W(FAKE)|", fake_wrd_cnt
        print "|P(REAL)|", prior_of_real
        print "|P(FAKE)|", prior_of_fake
        print "SMOOTHNING FACTOR (S)", SMOOTHNING_FACTOR
#         display(matrix)
        print "Predicting the class for new Sample now"
        new_sample = "October 31, 2016 at 4:52 am Pretty factual except for women in the selective service. American military is still voluntary only and hasn't been a draft since Vietnam war. The comment was made by a 4 star general of the army about drafting women and he said it to shut up liberal yahoos."
        print multinomial_NBC(
            new_sample, vocabulary, fake_wrd_cnt, real_wrd_cnt, matrix, prior_of_fake, prior_of_real)
    except IOError as err:
        print str(err)
    except UnicodeDecodeError as err:
        print str(err)
    except Exception as err:
        print str(err)

if __name__ == "__main__":
    main()
