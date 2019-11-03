#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Import doctor review dataset and gender dataset
path = "/Users/chongchen/Desktop/19Fall RA/raw data/"
doc_review = pd.read_csv(path + 'docs_before.csv', index_col = 0)
gender = pd.read_csv(path + 'genders.csv').drop_duplicates()


# preveiw two datasets
print(doc_review.head())
print(gender.head())


# Merge two datasets based on doctor id
doc_review = pd.merge(doc_review,gender,on='hp_id',how='inner')


# Diagnosis of null value
for i in doc_review.columns:
    a = doc_review[i].isna().sum()
    print('%s: %d' % (i,a))


# Average Helpness Comparison 
g = sns.FacetGrid(doc_review, row = 'gender', col = 'hasorder', sharex=True, sharey=True, height = 4)
g.map(sns.boxplot, 'avg_help', color = 'gray',flierprops={'marker': '.'})
# Male doctors' average helpness score is a little bit higher than that of female doctors'.


# Average knowledge score
g = sns.FacetGrid(doc_review, row = 'gender', col = 'hasorder', sharex=True, sharey=True, height = 4)
g.map(sns.boxplot, 'avg_know', color = 'cyan',flierprops={'marker': '.'})
# Here we could find something interesting: for doctors who haven't received sanctions, the average knowledge score of Male doctors are way larger than average knowledge score than female doctors compare with other score such as average helpness, average punctuality.

# verage punct score
g = sns.FacetGrid(doc_review, row = 'gender', col = 'hasorder', sharex=True, sharey=True, height = 4)
g.map(sns.boxplot, 'avg_punct', color = 'pink',flierprops={'marker': '.'})


# Get rid of the space after state abbr, speciaty
doc_review['HP_ST'] = doc_review['HP_ST'].map(lambda x: x[:2])

def no_space(x):
    a = x.split(' ')
    b = ''
    for i in a:
        if i != '':
            b = b+i
    return b

doc_review['spec_comb'] = doc_review['spec_comb'].map(lambda x:no_space(x))


# Doctor distribution across states
data_list = []
for i in doc_review['HP_ST'] .unique():
    data_list.append(doc_review[doc_review['HP_ST']==i]['avg_help'])

state_count = doc_review['HP_ST'].value_counts()
plt.figure(figsize=(20,10))
plt.bar(state_count.index, state_count.values)
plt.show()


# Doctor Specialty
plt.figure(figsize=(10,8))
doc_review['spec_comb'].value_counts().sort_values().plot.barh()


spec = doc_review['spec_comb'].value_counts()
dict = {}
for i in range(len(spec)):
    dict[spec.keys()[i]]=spec[i]


# Doctor specialty comparison between gender

doc_review.groupby(['gender']).spec_comb.value_counts().unstack(0).plot.barh()


# Doctor specialty comparison between 'hasorder'
doc_review.groupby(['hasorder']).spec_comb.value_counts().unstack(0).plot.barh()

