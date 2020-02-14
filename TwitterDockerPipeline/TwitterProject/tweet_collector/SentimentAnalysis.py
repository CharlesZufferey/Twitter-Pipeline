#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with a LSTM Neural Network

# In[2]:


import spacy
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import  Sequential
from keras import backend as K
from keras.callbacks import TensorBoard


# In[3]:


nlp = spacy.load('en_core_web_sm')

# ## Data Wrangling
# read the training data

# In[4]:


path = "data/train1 - Copy.tsv"
with open(path, 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    data = np.array(list(reader))


# In[5]:


xtrain = data[1:,:][:,2]
ytrain = data[1:,:][:,3]
ytrain_categorical = to_categorical(ytrain) # one-hot encoding ytrain


# In[6]:


# basic NLP: tokenization


# In[7]:


MAX_NUM_WORDS = 16000
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer.fit_on_texts(xtrain)
sequences = tokenizer.texts_to_sequences(xtrain)


# ### padding
# 
# too short sentences:
#     
# `a really good --> a really good PAD PAD PAD PAD PAD .. maxlenght`

# **TODO: try a smaller max length (10 tokens)**

# In[18]:


MAX_SEQUENCE_LENGTH = len(max(xtrain, key = len))
xtrain_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding = "post")


# In[34]:


word_index = tokenizer.word_index
EMBEDDING_DIM = 96 # vector length in spacy model


# ### Embedding
# 
# * look up all words from the vocabulary in spacy
# * build a huge lookup dictionary {'word': vector}

# In[35]:


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))


# In[23]:


for word, i in word_index.items():
    token = nlp(word)
    


# In[33]:


embedding_vector.shape
embedding_matrix.shape
embedding_matrix[1].shape


# In[36]:


for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    token = nlp(word)
    embedding_vector = token.vector  #getting vector form spacy
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# ### Model definition

# In[37]:


K.clear_session()
model = Sequential([
    Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], \
              input_length=MAX_SEQUENCE_LENGTH, trainable=False), # using Spacy weights
    SpatialDropout1D(0.2), # regularization
    LSTM(16, # neurons for interpreting sentence converted to Word Vectors
         # 16 neurons working in parallel
         # TUNE THIS HYPERPARAMETER
         recurrent_activation='hard_sigmoid', 
         use_bias=True, 
         kernel_initializer='glorot_uniform', 
         recurrent_initializer='orthogonal', 
         bias_initializer='zeros',
         dropout=0.2, # regularization
         recurrent_dropout=0.2),
    # produce output probabilities (mutually exclusive, add up to 1)
    Dense(5, activation = "softmax"),
])


# Loss functions:
# 
# * binary cross-entropy (logloss): binary classification
# * categorical cross-entropy: mutually exclusive multiclass classification (with softmax)
# * MSE - regression
# * other - read it in research papers

# In[38]:


# builds a computation graph
# here we could deploy on which CPU/GPU/server the model is running
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[40]:


# train for N iterations
iterations = 10 
model.fit(xtrain_padded, ytrain_categorical, epochs = iterations, validation_split=0.2, batch_size=1000, verbose = 1)


# In[242]:


#model.save('kaggle01.h5')


# In[ ]:


#model.load('kaggle01.h5')


# In[41]:


path = "data/tweets.csv"
with open(path, 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    #data = np.array(list(reader))
    data = list(reader)
print (data)
#print (data[1])

#%%
type(xtest)
#%%
#xtest = data[1:,:][:,2]
xtest = []
for i in data:
    xtest.append(i)
print (xtest)
# In[42]:


MAX_NUM_WORDS = 16000
# tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
# tokenizer.fit_on_texts(xtest)
#DONT RE-FIT THE TOKENIZER
sequences = tokenizer.texts_to_sequences(xtest)

#MAX_SEQUENCE_LENGTH = len(max(xtest, key = len))
xtest_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding = "post")


# In[43]:


preds = model.predict(xtest_padded)

print (preds)
# **TODO: check whether sentiment=2 is overrepresented in training data**

# In[44]:


preds[2]


# In[ ]:




