# Project_Veritas :bowtie:

This is project `Veritas`! It serves as an academic project for
CS5100: Foundation for Artificial Intelligence in Northeastern Univ.

## Contributors

Shubhi, Emily Dutile and Linghan Xing are the first contributors.

## What it does :rocket:

The project is an automated approach to identify authenticate news
from fake ones. 

## Approach

Use naive bayes classifier to tell Fake News vs Real News.


## Install
anaconda cloud (for jupyter notebooks)
scikit-learn
pandas

## Problem specification

Data representation:

Target: take our dataset and represent them in our datastructure.

Steps:

1. Extract the words: 

    * convert words into lower case, extract words

    * Apply stemming: reduce words to their root form: i.e. subscribed -> subscrib, 
    subscriber -> subscrib; in this case we could use **NLTK toolkit**.

2. Build a dictionary of vecabulary, only retain unique keywards

3. Vectorise document, loop over the dictionary and mark the frequency of each word.

    * term frequency (tf): boolean tf or raw count, or TF adjusted for length of d, or logarithmically scaled TF

    * inverse document frequency(IDF): IDF measures how rare the term is across all documents in the corpus

    * normalization after the tf-idf: L2 norm

## Files of Interest
	- /project/naiveBayes.py
	- /project/models_and_evals.ipynb
	- /project/tfidf_implementation.ipynb
	- /project/topicmodel.py


