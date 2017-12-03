import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from gensim import corpora,models
from collections import defaultdict
from datetime import datetime
import pyLDAvis.gensim
import warnings
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt


def process_text(text):
    tokens = tokenizer.tokenize(str(text).lower())
    stopped_tokens = [i for i in tokens if not i in stopWords]
    stopped_tokens = [word for word in stopped_tokens if word.isalpha()]
    stopped_tokens = bigram[stopped_tokens]
    tokens = [i for i in stopped_tokens]
    return tokens


def print_time():
    print("Current Time: " + str(datetime.now()))
    print("Time elapsed until now: " + str(datetime.now() - start_time))
    print("\n")


def generate_wordclouds(lda, topic_count,word_count):
    print("Latent Dirichlet Allocation......")
    print_time()
    print(lda.print_topics(-1, word_count))
    print_time()
    for i in range(topic_count):
        wordcloud.fit_words(dict(lda.show_topics(i+1,200,formatted=False)[0][1])).to_file('lda_all_'+str(i+1)+'.png')


def generate_ldavis(lda, topic_count, word_count):
    print("Latent Dirichlet Allocation......")
    print(lda.print_topics(-1, word_count))
    print_time()
    lda_vis = pyLDAvis.gensim.prepare(lda, doc_term_matrix, dictionary)
    pyLDAvis.save_html(lda_vis, 'visualization_all_' + str(topic_count) + '.html')


start_time = datetime.now()
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Start time of program: " + str(start_time))
tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))
wordcloud = WordCloud()
Lda = gensim.models.ldamodel.LdaModel

filename = "../data/fake_or_real_news.csv"
with open(filename,'rb') as f:
    lines = f.read()
new = str(lines,'utf-8')
with open('clear','w') as f2:
    f2.write(new)
df = pd.read_csv("clear")
df = df.set_index('Unnamed: 0')

bigram = gensim.models.phrases.Phrases(df.text)
df['text_tokens'] = df.text.apply(process_text)
doc_clean = df.text_tokens
frequency = defaultdict(int)

for text in doc_clean:
    for token in text:
        frequency[token] += 1

processed_corpus = [[token for token in text if frequency[token] > 10] for text in doc_clean]

dictionary = corpora.Dictionary(processed_corpus)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_corpus]

lda = models.LdaModel(doc_term_matrix, id2word=dictionary, num_topics=10, passes=20, iterations=50)

generate_wordclouds(lda, 10, 3)
generate_ldavis(lda, 10, 3)