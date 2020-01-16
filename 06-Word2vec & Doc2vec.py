#!/usr/bin/env python
# coding: utf-8

# ## Import doctor review dataset and gender dataset



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = "/Users/chongchen/Desktop/19Fall RA/data/generated_data/Internal Medicine Review.csv"
review_im = pd.read_csv(path, index_col = 0)
review_im.head()



# drop records without review
review_im.drop(review_im[review_im['Review'] == ' '].index, inplace = True)
review_im.reset_index(inplace = True, drop = True)

# check datatype
review_im['words'][0]
# convert list like string to real list
import ast
review_im['words'] = review_im['words'].map(lambda x: ast.literal_eval(x))
review_im['words'][0]


from gensim.models import Word2Vec
# preparing corpus
documents = list(review_im['words'])

import numpy as np
np.mean([len(i) for i in documents])
max([len(i) for i in documents])
min([len(i) for i in documents])

# see the distribution of review length, and then decide window size in word2vec
plt.hist([len(i) for i in documents], bins = 20, color = '0.25')




# sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
# sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, 
# hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, 
# batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None


# ### Hyperparameter setting 1
w2v = Word2Vec(size=300,window=30,min_count=30,workers=4,hs=0, negative=7)
w2v.build_vocab(documents)
w2v.train(documents, total_examples=w2v.corpus_count,epochs=100)



w2v.save("w2v_1.model")



w2v_1= Word2Vec.load("w2v_1.model")


# In[82]:


w2v_1.wv.most_similar(positive=["dr"])


# In[83]:


w2v_1.wv.most_similar(positive=["wonderful"])


# In[84]:


w2v_1.wv.most_similar(positive=["knowledgeable"])


# In[85]:


w2v_1.wv.most_similar(positive=["caring"])


# In[86]:


w2v_1.wv.most_similar(positive=["rude"])


# In[116]:


w2v_1.wv.most_similar(positive=["unprofessional"])


# In[113]:


w2v_1.wv.most_similar(positive=["staff"])


# In[223]:


w2v_1.wv.most_similar(positive=["he"])


# ### Hyperparameter setting 2 ( decrease window size from 30 to 20, holding the rest of the parameters constant)

# In[87]:


w2v = Word2Vec(size=300,window=20,min_count=30,workers=4,hs=0, negative=7)
w2v.build_vocab(documents)
w2v.train(documents, total_examples=w2v.corpus_count,epochs=100)


# In[88]:


w2v.save("w2v_2.model")


# In[89]:


w2v_2= Word2Vec.load("w2v_2.model")


# In[90]:


w2v_2.wv.most_similar(positive=["dr"])


# In[91]:


w2v_2.wv.most_similar(positive=["wonderful"])


# In[92]:


w2v_2.wv.most_similar(positive=["knowledgeable"])


# In[93]:


w2v_2.wv.most_similar(positive=["caring"])


# In[94]:


w2v_2.wv.most_similar(positive=["rude"])


# In[115]:


w2v_2.wv.most_similar(positive=["unprofessional"])


# In[112]:


w2v_2.wv.most_similar(positive=["staff"])


# ### Hyperparameter setting 3 ( increase window size to 40, holding the rest of the parameters constant)

# In[95]:


w2v = Word2Vec(size=300,window=40,min_count=30,workers=4,hs=0, negative=7)
w2v.build_vocab(documents)
w2v.train(documents, total_examples=w2v.corpus_count,epochs=100)


# In[96]:


w2v.save("w2v_3.model")


# In[97]:


w2v_3= Word2Vec.load("w2v_3.model")


# In[98]:


w2v_3.wv.most_similar(positive=["dr"])


# In[99]:


w2v_3.wv.most_similar(positive=["wonderful"])


# In[100]:


w2v_3.wv.most_similar(positive=["knowledgeable"])


# In[101]:


w2v_3.wv.most_similar(positive=["caring"])


# In[102]:


w2v_3.wv.most_similar(positive=["rude"])


# In[114]:


w2v_3.wv.most_similar(positive=["unprofessional"])


# In[111]:


w2v_3.wv.most_similar(positive=["staff"])


# ### Hyperparameter setting 4 ( change the window size to 35, holding the rest of the parameters constant)

# In[72]:


w2v = Word2Vec(size=300,window=35,min_count=30,workers=4,hs=0, negative=7)
w2v.build_vocab(documents)
w2v.train(documents, total_examples=w2v.corpus_count,epochs=100)


# In[78]:


w2v.save("w2v_4.model")


# In[103]:


w2v_4= Word2Vec.load("w2v_4.model")


# In[104]:


w2v_4.wv.most_similar(positive=["dr"])


# In[105]:


w2v_4.wv.most_similar(positive=["wonderful"])


# In[106]:


w2v_4.wv.most_similar(positive=["knowledgeable"])


# In[107]:


w2v_4.wv.most_similar(positive=["caring"])


# In[108]:


w2v_4.wv.most_similar(positive=["rude"])


# In[109]:


w2v_4.wv.most_similar(positive=["unprofessional"])


# In[110]:


w2v_4.wv.most_similar(positive=["staff"])


# ### Run Doc2vec - gender

# In[117]:


documents[0][0]


# In[118]:


gender = list(review_im['gender'])
hasorder = list(review_im['hasorder'])


# In[120]:


len(documents),len(gender),len(hasorder)


# In[123]:


# Tag document
import gensim
docs = [] # each review/row is a tagged document.
for i in range(len(documents)):
    docs.append(gensim.models.doc2vec.TaggedDocument(documents[i], [gender[i]]))


# In[124]:


docs[0]


# In[125]:


docs_for_vocab = [] 
for i in docs:
    docs_for_vocab.append(i)


# In[130]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import multiprocessing


# In[133]:


model = Doc2Vec(dm=0, window=30,vector_size=300,min_count=30,epochs=100,workers=4,
               hs=0,negative=7,dbow_words=1,dm_concat=1)
model.build_vocab(docs_for_vocab)
model.train(documents = docs,total_examples=model.corpus_count,epochs=100)


# In[134]:


model.save("d2v_g1.model")


# In[135]:


d2v_g1= Doc2Vec.load("d2v_g1.model")


# In[142]:


d2v_g1.wv.most_similar(positive = [d2v_g1.docvecs['F']],topn=15)


# In[224]:


d2v_g1.wv.most_similar(positive = [d2v_g1.docvecs['F']],topn=100)


# In[143]:


d2v_g1.wv.most_similar(positive = [d2v_g1.docvecs['M']],topn=15)


# ### Run Doc2vec - gender & sanction

# In[144]:


# Tag document
import gensim
docs_gs = [] # each review/row is a tagged document.
for i in range(len(documents)):
    docs_gs.append(gensim.models.doc2vec.TaggedDocument(documents[i],  [hasorder[i],gender[i]]))


# In[145]:


docs_gs[0]


# In[146]:


docs_for_vocab_gs = [] 
for i in docs_gs:
    docs_for_vocab_gs.append(i)


# In[147]:


model_gs = Doc2Vec(dm=0, window=30,vector_size=300,min_count=30,epochs=100,workers=4,
               hs=0,negative=7,dbow_words=1,dm_concat=1)
model_gs.build_vocab(docs_for_vocab_gs)
model_gs.train(documents = docs_gs,total_examples=model_gs.corpus_count,epochs=100)


# In[148]:


model_gs.save("d2v_gs1.model")


# In[149]:


d2v_gs1= Doc2Vec.load("d2v_gs1.model")


# In[158]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[0],d2v_gs1.docvecs['F']],topn=15)


# In[159]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[0],d2v_gs1.docvecs['M']],topn=15)


# In[160]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[1],d2v_gs1.docvecs['F']],topn=15)


# In[225]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[1],d2v_gs1.docvecs['F']],topn=50)


# In[161]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[1],d2v_gs1.docvecs['M']],topn=15)


# In[164]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[0]],topn=15)


# In[165]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs[1]],topn=15)


# In[166]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs['F']],topn=15)


# In[167]:


d2v_gs1.wv.most_similar(positive = [d2v_gs1.docvecs['M']],topn=15)


# ### Run Doc2vec - gender & sanction (downsampling)

# In[191]:


#1000 sanction male/female reviews and 1000 not sanction male/female reviews
import random
repeat = {}
indices = []
batch = []
draw = 1000
sf_candidates = list(review_im[np.logical_and(review_im['hasorder']==1, review_im['gender']=='F')].index)
sm_candidates = list(review_im[np.logical_and(review_im['hasorder']==1, review_im['gender']=='M')].index)
nsf_candidates = list(review_im[np.logical_and(review_im['hasorder']==0, review_im['gender']=='F')].index)
nsm_candidates = list(review_im[np.logical_and(review_im['hasorder']==0, review_im['gender']=='M')].index)
for b in range(draw):
    sf = random.choice(sf_candidates)
    sm = random.choice(sm_candidates)
    indices.append(sf)
    indices.append(sm)
    nsf = random.choice(nsf_candidates)
    nsm = random.choice(nsm_candidates)
    indices.append(nsf)
    indices.append(nsm)
for i in indices:
    try:
        repeat[i]+=1
    except:
        repeat[i]=1
    batch.append(docs_gs[i])


# In[192]:


batch[1]


# In[193]:


model_gs2 = Doc2Vec(dm=0, window=30,vector_size=300,min_count=30,epochs=100,workers=4,
               hs=0,negative=7,dbow_words=1,dm_concat=1)
model_gs2.build_vocab(docs_for_vocab_gs)
model_gs2.train(documents = batch,total_examples=model_gs2.corpus_count,epochs=100)


# In[194]:


model_gs2.save("d2v_gs2.model")


# In[195]:


d2v_gs2= Doc2Vec.load("d2v_gs2.model")


# In[196]:


d2v_gs2.wv.most_similar(positive = [d2v_gs2.docvecs[0],d2v_gs2.docvecs['F']],topn=15)


# In[197]:


d2v_gs2.wv.most_similar(positive = [d2v_gs2.docvecs[0],d2v_gs2.docvecs['M']],topn=15)


# In[198]:


d2v_gs2.wv.most_similar(positive = [d2v_gs2.docvecs[1],d2v_gs2.docvecs['F']],topn=15)


# In[199]:


d2v_gs2.wv.most_similar(positive = [d2v_gs2.docvecs[1],d2v_gs2.docvecs['M']],topn=15)


# ### Run Doc2vec - gender & sanction (upsampling)

# In[204]:


#1000 sanction male/female reviews and 1000 not sanction male/female reviews
import random
repeat = {}
indices = []
batch2 = []
draw = 10000
sf_candidates = list(review_im[np.logical_and(review_im['hasorder']==1, review_im['gender']=='F')].index)
sm_candidates = list(review_im[np.logical_and(review_im['hasorder']==1, review_im['gender']=='M')].index)
nsf_candidates = list(review_im[np.logical_and(review_im['hasorder']==0, review_im['gender']=='F')].index)
nsm_candidates = list(review_im[np.logical_and(review_im['hasorder']==0, review_im['gender']=='M')].index)
for b in range(draw):
    sf = random.choice(sf_candidates)
    sm = random.choice(sm_candidates)
    indices.append(sf)
    indices.append(sm)
    nsf = random.choice(nsf_candidates)
    nsm = random.choice(nsm_candidates)
    indices.append(nsf)
    indices.append(nsm)
for i in indices:
    try:
        repeat[i]+=1
    except:
        repeat[i]=1
    batch2.append(docs_gs[i])


# In[205]:


len(batch2)


# In[206]:


model_gs3 = Doc2Vec(dm=0, window=30,vector_size=300,min_count=30,epochs=100,workers=4,
               hs=0,negative=7,dbow_words=1,dm_concat=1)
model_gs3.build_vocab(docs_for_vocab_gs)
model_gs3.train(documents = batch2,total_examples=model_gs2.corpus_count,epochs=100)


# In[207]:


model_gs3.save("d2v_gs3.model")


# In[208]:


d2v_gs3= Doc2Vec.load("d2v_gs3.model")


# In[209]:


d2v_gs3.wv.most_similar(positive = [d2v_gs3.docvecs[0],d2v_gs3.docvecs['F']],topn=15)


# In[210]:


d2v_gs3.wv.most_similar(positive = [d2v_gs3.docvecs[0],d2v_gs3.docvecs['M']],topn=15)


# In[211]:


d2v_gs3.wv.most_similar(positive = [d2v_gs3.docvecs[1],d2v_gs3.docvecs['F']],topn=15)


# In[226]:


d2v_gs3.wv.most_similar(positive = [d2v_gs3.docvecs[1],d2v_gs3.docvecs['F']],topn=50)


# In[212]:


d2v_gs3.wv.most_similar(positive = [d2v_gs3.docvecs[1],d2v_gs3.docvecs['M']],topn=15)


# In[227]:


d2v_gs3.wv.most_similar(positive = [d2v_gs3.docvecs[1],d2v_gs3.docvecs['M']],topn=50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Run Doc2vec - sanction

# In[213]:


# Tag document
import gensim
docs_s = [] # each review/row is a tagged document.
for i in range(len(documents)):
    docs_s.append(gensim.models.doc2vec.TaggedDocument(documents[i],  [hasorder[i]]))


# In[214]:


docs_s[0]


# In[215]:


docs_for_vocab_s = [] 
for i in docs_s:
    docs_for_vocab_s.append(i)


# In[216]:


model_s = Doc2Vec(dm=0, window=30,vector_size=300,min_count=30,epochs=100,workers=4,
               hs=0,negative=7,dbow_words=1,dm_concat=1)
model_s.build_vocab(docs_for_vocab_s)
model_s.train(documents = docs_s,total_examples=model_s.corpus_count,epochs=100)


# In[217]:


model_s.save("d2v_s1.model")


# In[218]:


d2v_s1= Doc2Vec.load("d2v_s1.model")


# In[219]:


d2v_s1.wv.most_similar(positive = [d2v_s1.docvecs[0]],topn=15)


# In[220]:


d2v_s1.wv.most_similar(positive = [d2v_s1.docvecs[1]],topn=15)


# 1. Try window size: 5
# 2. Take the absolute difference, for gender, sanction and both
# 3. Try one feature
