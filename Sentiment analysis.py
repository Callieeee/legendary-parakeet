#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the doctor review dataset
path = "/Users/chongchen/Desktop/19Fall RA/raw data/"
doc_review = pd.read_csv(path + 'docs_before.csv', index_col = 0)
gender = pd.read_csv(path + 'genders.csv').drop_duplicates()
doc_review = pd.merge(doc_review,gender,on='hp_id',how='inner')


# Number of records and number of columns
print(doc_review.shape)

# Column Names
print(doc_review.columns)


doc_review['hasorder'] = doc_review['hasorder'].map(lambda x: str(x))
doc_review['hasorder'].value_counts()
# 1371 out of 133600 doctors have received sanctions.


def no_space(x):
    a = x.split(' ')
    b = ''
    for i in a:
        if i != '':
            b = b+i
    return b
doc_review['spec_comb'] = doc_review['spec_comb'].map(lambda x:no_space(x))


# Let's focus on the doctors who has speciality of 'Internal Medicine' first.
review = doc_review[['hp_id','review_corpus']][doc_review['spec_comb']=='InternalMedicine']
print(review.shape)

review.reset_index(inplace = True, drop = True)


# Convert the review corpus into a list of sentences
review['sentence']= review['review_corpus'].map(lambda x: x.split('|'))


# Create a function to generate a new dataframe of doc's review by seperating review corpus
review_df = pd.DataFrame()
for i, sentence in enumerate(review['sentence']):
    temp_dict = dict(enumerate(sentence))
    a = len(list(temp_dict.keys()))
    s = str(review['hp_id'][i])
    temp_df = pd.DataFrame.from_dict(data = temp_dict, orient = 'index', columns=['Review'])
    temp_df['hp_id'] = [s for i in range(a)]
    review_df = review_df.append(temp_df, ignore_index=True)
# The new dataframe with each review in a single cell.
print(review_df)

# Further split each sentence into a list of words
from nltk.tokenize import word_tokenize
review_df['word'] = review_df['Review'].map(lambda sentence: word_tokenize(sentence))
print(review_df['Review'][0])
print(review_df['word'][0])



# Read NRC-Emotional Lexicon
# This NRC-Emotional Lexicon will be used to detect 8 kinds of emotions of each review
list_e = [line.split('\t') for line in open('/Users/chongchen/Desktop/19Fall RA/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')]

for i in list_e[:10]:
    i[2] = i[2].strip('\n')
    print(i)


nrc_dict = {}
for i in list_e:
    if i[0] not in nrc_dict.keys():
               nrc_dict[i[0]] = {i[1]:int(i[2].strip('\n'))}
    elif i[0] in nrc_dict.keys():
               nrc_dict[i[0]][i[1]] = int(i[2].strip('\n'))
print(nrc_dict)


# Read the AFINN dictionary 
# This AFINN dictionary will be used to calculated the sentiment score of each review
list_a = [line.split('\t') for line in open("/Users/chongchen/Desktop/19Fall RA/AFINN/AFINN-111.txt")]
afinn = dict(map(lambda x : [x[0],int(x[1])], list_a))
print(afinn)


# For example, the sentiment score of 'good' is 3.
afinn["Good".lower()]

# The sentiment score of "Rainy day but still in a good mood" is 2 based on AFINN dictionary.
print(sum(map(lambda word: afinn.get(word, 0), "Rainy day but still in a good mood".lower().split())))

# Caculate the sentiment score of each review
# Sentiment score = total score / number of words contribute to the total score. Since the more words, the higher the score will be, this average score try to eliminate the effect of long comments.

# calculate total score of each review
review_df['score'] = review_df['word'].map(lambda x: sum(afinn.get(word, 0) for word in x) )


# define a function to calcualte the average score
def avg_score(sentence):
    num = 0
    score = 0
    for word in sentence:
        s = afinn.get(word, 0)
        score = score+s
        if s != 0:
            num = num+1
    if num == 0:
        final_score = 0
    else:
        final_score = score/num
    return final_score


# calculate the average score of each comment
review_df['avg_score'] = review_df['word'].map(lambda x: avg_score(x))
review_df.sort_values(by=['avg_score','score'], axis=0, ascending = False, inplace = True)
review_df.reset_index(drop = True, inplace = True)

print(review_df['Review'][0],'\n')
print('Total Score: ',review_df['score'][0],'\n' )
print('Average Score: ',review_df['avg_score'][0],'\n' )


# Statistic of Score

# Total Score
print(review_df['score'].describe())


# Average Score
review_df['avg_score'].describe()


# Let's look at the following ramdom selected 10 sentences to see the accuracy of the results
from numpy.random import randint

nums = randint(0,89661,10)
for i in randint(0,89661,10):
    print('\n', 'No.',i+1)
    print('Content: ',review_df['Review'][i],'\n')
    print('Total Score: ', review_df['score'][i])
    print('Average Score: ', review_df['avg_score'][i], '\n')


# Add Gender information to the current data
gender['hp_id'] = gender['hp_id'].map(lambda x: str(x))
review_df = pd.merge(review_df, gender, on='hp_id', how = 'left')
print(review_df.head())


# Some statistics of sentiment score
# Statistics of male doctors total sentiment score
print(review_df['score'][review_df['gender']=='M'].describe())


# Statistics of female doctors total sentiment score
print(review_df['score'][review_df['gender']=='F'].describe())


# Statistics of male doctors average sentiment score
print(review_df['avg_score'][review_df['gender']=='M'].describe())


# Statistics of female doctors average sentiment score
print(review_df['avg_score'][review_df['gender']=='F'].describe())


# Comparing distribution of total sentiment score between gender
g=sns.FacetGrid(review_df, col='gender',sharex=True, height=4)
g.map(plt.hist, 'score', color ='green')


# Comparing distribution of average sentiment score between gender
g=sns.FacetGrid(review_df, col='gender',sharex=True, height=4)
g.map(plt.hist, 'avg_score', color ='blue')


plt.figure(figsize=(8,6))
plt.hist(review_df['score'][review_df['gender']=='M'], label = 'Male', density= True, alpha = 0.5)
plt.hist(review_df['score'][review_df['gender']=='F'], label = 'Female', density = True, alpha = 0.75)
plt.legend()
plt.xticks(np.arange(-50,51,5))
plt.title('Comparing Distribuition of Total Score between Genders (Normalized)', fontsize = 12, loc = 'center')


plt.figure(figsize=(8,6))
plt.hist(review_df['avg_score'][review_df['gender']=='M'], label = 'Male', density= True, alpha = 0.25)
plt.hist(review_df['avg_score'][review_df['gender']=='F'], label = 'Female', density = True, alpha = 0.5)
plt.legend()
plt.xticks(np.arange(-5,6,1))
plt.title('Comparing Distribuition of Average Score between Genders (Normalized)', fontsize = 12, loc = 'center')

# add sanction information
doc_review['hp_id'] = doc_review['hp_id'].map(lambda x: str(x))
sanction = doc_review[['hp_id','hasorder']]
review_df = pd.merge(review_df,sanction, on = 'hp_id', how = 'left')


# Calculate the average score of each doctor

# Average Score = sum(avg_score per review)/(num of reviews with score higher than 0)
score_df = review_df.groupby(by=['hp_id'])['avg_score'].sum()
score_df2 = review_df.groupby(by=['hp_id'])['Review'].count()
score_df3 = review_df[review_df['score']!=0].groupby(by=['hp_id'])['Review'].count()
score_df = pd.merge(score_df, score_df2, on='hp_id', how = 'inner')
score_df = pd.merge(score_df, score_df3, on='hp_id', how = 'left')
score_df = pd.merge(score_df, doc_review[['hp_id','gender','hasorder']], on='hp_id')
score_df.rename(mapper={'Review_x':'Total_num of Reviews', 'Review_y':'Total_num of reviews used', 'avg_score':'total_avg_score'}, axis=1, inplace = True)
score_df['avg_score'] = score_df['total_avg_score']/score_df['Total_num of reviews used']
print(score_df.head())


# Statistics of avg_score of doctors who received sanctions 
score_df['avg_score'][score_df['hasorder']=='1'].describe()


# Statistics of avg_score of doctors who never receive sanctions 
score_df['avg_score'][score_df['hasorder']=='0'].describe()


# Statistics of avg_score of male doctors
score_df['avg_score'][score_df['gender']=='M'].describe()


# Statistics of avg_score of female doctors
score_df['avg_score'][score_df['gender']=='F'].describe()


plt.figure(figsize= (8,6))
plt.hist(score_df['avg_score'][score_df['hasorder']=='0'], label = 'No sanction', color = 'yellow')
plt.title('Avg_score Distribution of \'NotOrdered\' doctors', fontsize = 12, loc = 'center')



plt.figure(figsize= (8,6))
plt.hist(score_df['avg_score'][score_df['hasorder']=='1'], label = 'Has sanction',color = 'green')
plt.title('Avg_score Distribution of \'Hasorder\' doctors', fontsize = 12, loc = 'center')
plt.xticks(np.arange(-4,5,1))



plt.figure(figsize=(12,8))
plt.hist(score_df['avg_score'][score_df['hasorder']=='0'], label = 'No sanction', color = 'yellow', density=True,alpha=0.75)
plt.hist(score_df['avg_score'][score_df['hasorder']=='1'], label = 'Has sanction', color = 'green', density=True, alpha=0.5)
plt.legend()
plt.xticks(np.arange(-5,5,1))
plt.title('Score Distribuition Comparison between \'Hasorder\' (Normalized)', fontsize = 12, loc = 'center')


score_df['hasorder'] = score_df['hasorder'].map(lambda x: str(x))


# Normalized distribution of average score comparison between 'hasorder' and 'gender
g2 = sns.FacetGrid(score_df, row='hasorder', col='gender',sharex=True)
g2.map(sns.distplot, 'avg_score', rug=True)



df3 = pd.crosstab(index = score_df.gender, columns = score_df.hasorder)
df3['percentage'] = df3['1']/df3['0']
print(df3)
