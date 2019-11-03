#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from IPython.display import Image  
import pydotplus
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[2]:


review_feature = pd.read_csv("/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal Medicine_review_features.csv", index_col = 0)


# In[3]:


review_feature.head()


# In[4]:


review_feature.columns


# In[5]:


# 1 for male, 0 for female
review_feature['gender'] = review_feature['gender'].map(lambda x: 1 if x == 'M' else 0)


# In[6]:


review_feature['avg_staff'].fillna(0, inplace = True)


# In[7]:


x = review_feature[['Number of Review', 'averaeg length', 'avg_help',
       'avg_know', 'avg_punct', 'avg_staff', 'anger', 'anticipation',
       'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise',
       'trust']]
y_gender = review_feature['gender']
y_hasorder = review_feature['hasorder']


# ### Predict gender

# In[12]:


import statsmodels.api as sm
logit_model=sm.Logit(y_gender,x)
result=logit_model.fit()
print(result.summary2())


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y_gender, test_size=0.3, random_state=0)


# In[14]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[22]:


y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(x_test, y_test)))
print('Baseline accuracy on test data set: {:.4f}'.format(pd.Series(y_test).value_counts()[1]/np.sum([1 for x in y_test])))


# In[16]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Try Classification Tree

# In[49]:


# Create decision tree classifer object
clf = DecisionTreeClassifier(random_state=0,max_leaf_nodes=10)

# Train model
model = clf.fit(x_train, y_train)


# In[50]:


y_predict = clf.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(clf.score(x_test, y_test)))
print('Baseline accuracy on test data set: {:.4f}'.format(pd.Series(y_test).value_counts()[1]/np.sum([1 for x in y_test])))


# In[51]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix)


# In[58]:





# In[61]:


# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=list(x_train.columns),  
                                class_names=['0','1'])
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())


# In[63]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fpr1, tpr1, thresholds1 = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
clf_roc_auc = roc_auc_score(y_test, clf.predict(x_test))
fpr2, tpr2, thresholds2 = roc_curve(y_test, clf.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr1, tpr1, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.plot(fpr2, tpr2, label='Classification Tree (area = %0.2f)' % clf_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ### Predict 'has order'

# In[65]:


import statsmodels.api as sm
logit_model=sm.Logit(y_hasorder,x)
result=logit_model.fit()
print(result.summary2())


# In[66]:


x1_train, x1_test, y1_train, y1_test = train_test_split(x, y_hasorder, test_size=0.3, random_state=3)


# In[67]:


logreg = LogisticRegression()
logreg.fit(x1_train, y1_train)
y1_pred = logreg.predict(x1_test)
print('Accuracy of logistic regression classifier on test set: {:.6f}'.format(logreg.score(x1_test, y1_test)))
print('Baseline accuracy: {:.6f}'.format(pd.Series(y1_test).value_counts()[0]/np.sum([1 for x in y1_test])))


# In[68]:


print(classification_report(y1_test, y1_pred))


# Classification Tree

# In[181]:


# Create decision tree classifer object
clf2 = DecisionTreeClassifier(random_state=0,max_leaf_nodes=8,class_weight={1:2})

# Train model
model2 = clf2.fit(x1_train, y1_train)


# In[182]:


y_predict = clf2.predict(x1_test)
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(clf2.score(x1_test, y1_test)))
print('Baseline accuracy on test data set: {:.4f}'.format(pd.Series(y1_test).value_counts()[0]/np.sum([1 for x in y1_test])))


# In[183]:


# Create DOT data
dot_data = tree.export_graphviz(clf2, out_file=None,
                                feature_names=list(x1_train.columns),  
                                class_names=['0','1'])
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())


# In[ ]:





# In[184]:


logit_roc_auc = roc_auc_score(y1_test, logreg.predict(x1_test))
clf2_roc_auc = roc_auc_score(y1_test, clf2.predict(x1_test))
fpr1, tpr1, thresholds1 = roc_curve(y1_test, logreg.predict_proba(x1_test)[:,1])
fpr2, tpr2, thresholds2 = roc_curve(y1_test, clf2.predict_proba(x1_test)[:,1])
plt.figure()
plt.plot(fpr1, tpr1, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr2, tpr2, label='Classification Tree (area = %0.2f)' % clf2_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

