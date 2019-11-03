#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Read the doctor review dataset
review_df = pd.read_csv("/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal_Medince_Review_with_emotion.csv", index_col = 0)
emotion_df = review_df.groupby(by = 'hp_id')['hp_id','anger', 'anticipation', 'disgust', 'fear',
       'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust'].count()
print(emotion_df)
emotion_df.reset_index(inplace= True)

# calcualte emotion score for each doctor
emotion_score = review_df.groupby(by = 'hp_id').sum()
emotion_score.reset_index(inplace= True)



emotion_list = ['anger', 'anticipation',
       'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise',
       'trust']

for i in emotion_list:
    a = np.array(emotion_score[i])
    b = np.array(emotion_df[i])
    emotion_df[i] = np.nan_to_num(a/b)

print(emotion_df)
emotion_df.to_csv("/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal_Medicine_Average_emotion_score.csv")


path = "/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal_Medicine_Summarize_Review.csv"
review_summarize = pd.read_csv(path)
review_summarize.drop(columns = ['Unnamed: 0'], inplace = True)


review_feature = review_summarize[['hp_id','gender','Number of Review','averaeg length','avg_help', 'avg_know', 'avg_punct', 'avg_staff']]
review_feature = pd.merge(review_feature,emotion_df, on = 'hp_id')

review = pd.read_csv("/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal_Medicine_Summarize_Review.csv")
review_hasorder = review[['hp_id','hasorder']]

review_feature = pd.merge(review_feature,review_hasorder, on='hp_id')


print(review_feature)
review_feature.to_csv("/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal Medicine_review_features.csv")



