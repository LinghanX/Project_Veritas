import math
import os
import json
import pandas as pd
import nltk.data
import nltk.tokenize
import nltk.stem
from nltk.corpus import stopwords

from IPython.display import display

SMOOTHNING_FACTOR = 1
world_label_matrix = []
NUM_OF_LABELS = 2
# Traning data consists of 70 % of the original
# input file
TRAINING_SPLIT_PERCENT = 0.80
DATA_FILE = "../data/small_dataset.csv"
TRAINING_FILE = "../data/training_naive_bayes.json"

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
        vocabulary.update([word for word in doc if word != "text"])
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

def prob_of_wrd_gvn_class(
    vocabulary, fake_word_count, real_word_count, label_word_matrix, word):
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

def evaluateForAccuracy(testingDF, prior_of_fake, prior_of_real):
    """
    Testing dataFrame will get generated from the original
    Dataframe. Which constitutes the last 1901 rows
    of the original Dataframe
    """
    results = []
    trained_file = open(TRAINING_FILE, "r")
    trainingData = json.loads(trained_file.read())
    fake_word_count = 0
    real_word_count = 0
    total_words = 0
    trainingDataDict = {}

    # calculate the essentials from the trained dataset
    for word, freq in trainingData:
        total_words+=1
        fake, real = freq
        fake_word_count+=int(fake)
        real_word_count+=int(real)
        trainingDataDict[word] = (
            int(fake), int(real))

    # getLabels for each row in the testDataFrame
    # calculates the accuracy count for predicted labels
    accuracyCount = 0
    snowball = nltk.stem.snowball.EnglishStemmer()
    for index, row in testingDF.iterrows():
        words = extract_words(row["text"], snowball, True)
        product_gvn_fake = prior_of_fake
        product_gvn_real = prior_of_real
        label = None
        for word in words:
            freq_tuple = trainingDataDict.get(word, None)
            fake_freq, real_freq = freq_tuple if freq_tuple else (0, 0)
            prob_of_wrd_gvn_fake = float(
                fake_freq + SMOOTHNING_FACTOR)/float(
                fake_word_count + total_words)
            prob_of_wrd_gvn_real = float(
                real_freq + SMOOTHNING_FACTOR)/float(
                real_word_count + total_words)
            product_gvn_fake*=prob_of_wrd_gvn_fake
            product_gvn_real*=prob_of_wrd_gvn_real
        if product_gvn_real > product_gvn_fake:
            label = "REAL"
        else:
            label = "FAKE"
        if label == row["label"].strip().upper():
            accuracyCount+=1
        results.append(label)
    # Number of rows labeled accuratly/ Total number of rows
    print "Accuracy %",float(accuracyCount)/float(len(testingDF))*100
    return results


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

def get_priors(dataDF):
    label_priors = calculate_prior_probabilities(dataDF)
    prior_of_real = float(label_priors["REAL"])/float(label_priors["REAL"] + label_priors["FAKE"])
    prior_of_fake = float(label_priors["FAKE"])/float(label_priors["REAL"] + label_priors["FAKE"])
    return prior_of_real, prior_of_fake

def main():
    try:
        dataDF_1 = pd.read_csv(
            DATA_FILE,
            sep=',', lineterminator='\n',
            names = ["title", "text", "label"], encoding="utf-8")

        # dataDF is used for training
        dataDF = dataDF_1.copy()
        dataDF = dataDF_1.head(
            int(TRAINING_SPLIT_PERCENT * len(dataDF_1)))

        #testingDF is used for testing
        testingDF = dataDF_1.copy()
        testingDF = testingDF.iloc[
            int(TRAINING_SPLIT_PERCENT * len(dataDF_1)):]

        print "Num of Rows in Source:Training:Test", len(dataDF_1), len(dataDF), len(testingDF)

        print "Stemming, stopwords removal from data"
        snowball = nltk.stem.snowball.EnglishStemmer()
        dataDF["Transformed text"] = dataDF.apply(
            lambda row: extract_words(row['text'], snowball, True), axis=1)

        print "Building vocabulary"
        vocabulary = build_vocabulary(dataDF["Transformed text"])

        print "Calculating priors"
        label_priors = calculate_prior_probabilities(dataDF)
        prior_of_real = float(label_priors["REAL"])/float(label_priors["REAL"] + label_priors["FAKE"])
        prior_of_fake = float(label_priors["FAKE"])/float(label_priors["REAL"] + label_priors["FAKE"])

        print "Building Label Word count Matrix"
        matrix = label_word_count_matrix(vocabulary, dataDF)

        print "Writing Trained Model to file"
        f = open(TRAINING_FILE, "w")
        f.write(json.dumps(zip(vocabulary, matrix)))
        f.close()

        print "Calculating Accuracy Results"
        results = evaluateForAccuracy(testingDF, prior_of_fake, prior_of_real)

        print "Writing resulting labels to file"
        f = open("../data/results_nb.json", "w")
        f.write(json.dumps(results))
        f.close()
    except IOError as err:
        print str(err)
    except UnicodeDecodeError as err:
        print str(err)
    except Exception as err:
        print str(err)

if __name__ == "__main__":
    main()
