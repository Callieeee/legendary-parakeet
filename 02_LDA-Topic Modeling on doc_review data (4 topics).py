#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/chongchen/Desktop/19Fall RA/raw data/"
doc_review = pd.read_csv(path + 'docs_before.csv', index_col = 0)
gender = pd.read_csv(path + 'genders.csv').drop_duplicates()
doc_review = pd.merge(doc_review,gender,on='hp_id',how='inner')

print(doc_review.shape)
print(doc_review.columns)

# Get rid of the space before doctor specialty
def no_space(x):
    a = x.split(' ')
    b = ''
    for i in a:
        if i != '':
            b = b+i
    return b

doc_review['spec_comb'] = doc_review['spec_comb'].map(lambda x:no_space(x))

# Let's focus on Internal Medicine doctors first
review = doc_review[['hp_id','review_corpus']][doc_review['spec_comb']=='InternalMedicine']
print(review.shape)

# reset index
review.reset_index(inplace = True, drop = True)


# Convert the review corpus into a list of sentences
review['sentence']= review['review_corpus'].map(lambda x: x.split('|'))


# Further split each sentence into a list of words
from nltk.tokenize import word_tokenize
review['word'] = review['sentence'].map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
print(review['word'][0][:3])


# Lemmatize with POS Tag

# reference : https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet


review['word_pos_tag'] = review['word'].map(lambda x: [pos_tag(i) for i in x])
print(review['word_pos_tag'][0][:3])

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = word.upper()[0]
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


for word in review['word_pos_tag'][0][3]:
    print(word[0])
    print(word[1])


# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

review['lemmatize_word'] = review['word_pos_tag'].map(
    lambda sentence_list: 
    [ [ lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) if word[0] not in [',','.'] else word[0] for word in sentence] 
        for sentence in sentence_list]
)

print(review['lemmatize_word'][0][:3])


word = 'caring'
print(lemmatizer.lemmatize(word))
print(lemmatizer.lemmatize(word,'v'))


# Remove stop words and combine tokens from all comments on a doctor

from nltk.corpus import stopwords
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
stopwords_other = ['one', 'Dr.', 'Doctor', 'doctor','bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
my_stopwords = stopwords.words('English') + stopwords_verbs + stopwords_other

from itertools import chain

review['tokens'] = review['lemmatize_word'].map(lambda x: list(chain.from_iterable(x)))
review['tokens'] = review['tokens'].map(lambda x: [i.lower() for i in x if i.isalpha() and i.lower() not in my_stopwords and len(i) >1])
print(review['tokens'][0][:20])


# Prepare bi-grams and tri-grams¶
# Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
# reference: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

tokens = review['tokens'].tolist()
bigram = gensim.models.Phrases(tokens)
trigram = gensim.models.Phrases(bigram[tokens], min_count = 1) 
tokens = list(trigram[bigram[tokens]])


# Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary
dictionary_lda = corpora.Dictionary(tokens)
dictionary_lda.filter_extremes(no_below=3)


# Term Document Frequency
corpus = [dictionary_lda.doc2bow(token) for token in tokens]


# Human readable format of corpus (term-frequency)
[[(dictionary_lda[id], freq) for id, freq in cp] for cp in corpus[:1]]


# Building the Topic Model
import numpy as np

# Build LDA model
np.random.seed(123456)
lda_model = gensim.models.ldamodel.LdaModel(corpus,num_topics = 4, id2word = dictionary_lda, passes = 6, alpha='auto',eta=[0.01]*len(dictionary_lda.keys()))

# Print the Keyword in the 4 topics
from pprint import pprint
pprint(lda_model.print_topics())


# How to interpret this?
# 
# Topic 0 is a represented as '0.131*"doc" + 0.091*"good" + 0.070*"great" + 0.042*"guy" + 0.019*"love" + '
#   '0.017*"thing" + 0.015*"spend_lot_time" + 0.015*"bedside_manner" + 0.015*"dr" + 0.014*"bed_side_manner".
# 
# It means the top 10 keywords that contribute to this topic are: ‘doc’, ‘good’, ‘great’.. and so on and the weight of ‘good’ on topic 0 is 0.091.
# 
# The weights reflect how important a keyword is to that topic.

doc_lda = lda_model[corpus]

print(lda_model[corpus[1]])
print(lda_model[corpus[2]])
print(lda_model[corpus[3]])
print(lda_model[corpus[5]])


# Visualize the topics-keywords
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary_lda)
vis


# reference: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# 
# So how to infer pyLDAvis’s output?
# 
# Each bubble on the left-hand side plot represents a topic. The larger the bubble, the more prevalent is that topic.
# 
# A good topic model will have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant.
# 
# A model with too many topics, will typically have many overlaps, small sized bubbles clustered in one region of the chart.
# 
# Alright, if you move the cursor over one of the bubbles, the words and bars on the right-hand side will update. These words are the salient keywords that form the selected topic.
# 
# We have successfully built a good looking topic model.
# 
# Given our prior knowledge of the number of natural topics in the document, finding the best model was fairly straightforward.
# 
# Upnext, we will improve upon this model by using Mallet’s version of LDA algorithm and then we will focus on how to arrive at the optimal number of topics given any large corpus of text.



# Let's look at the dominant topic of each review corpus of each doctor
review['TDF'] = review['tokens'].map(lambda x: dictionary_lda.doc2bow(x))

# initiate a dataframe
topics_df = pd.DataFrame()


def get_topic(token):
    topic_dict = dict(lda_model[token])
    for i in topic_dict.keys():
        if topic_dict[i] == max(topic_dict.values()):
            topic = i
    return topic

topics_df['topic'] = review['TDF'].map(lambda x: get_topic(x))

def get_percent_contrib(token):
    topic_dict = dict(lda_model[token])
    percent = max(topic_dict.values())
    return percent

topics_df['topic_percent_contrib'] = review['TDF'].map(lambda x: get_percent_contrib(x))
topics_df['Doc_Id'] = review['hp_id']
topics_df['review_text'] = review['review_corpus']


def get_keywords(x):
    topic_keywords = ','.join([word for word, prop in lda_model.show_topic(x)])
    return topic_keywords


topics_df['topic_keywords'] = topics_df['topic'].map(lambda x: get_keywords(x))
topics_df = topics_df[['Doc_Id', 'review_text','topic', 'topic_percent_contrib', 'topic_keywords']]
print(topics_df)

