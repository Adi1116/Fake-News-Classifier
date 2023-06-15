# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:33:34 2023

@author: adity
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

df = pd.read_csv("E:\ANACONDA\Fake News.csv")
df

df.describe()

df = df.dropna()

x = df.drop('label', axis = 1)
y = df['label']

print(x.shape)
print(y.shape)

fake = x.copy()
fake.reset_index(inplace = True)

fake['title'][0]

ps = PorterStemmer()
corpus = []
for i in range(0, len(fake)):
    review = re.sub('[^a-zA-Z]', ' ', str(fake['title'][i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
voc_size = 1000

onehot_rep = [one_hot(words, voc_size) for words in corpus]


sent_length = 20

emb_doc = pad_sequences(onehot_rep, padding = 'pre', maxlen = sent_length)

emb_vec_feat = 40
model = Sequential()
model.add(Embedding(voc_size, emb_vec_feat, input_length = sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

x_final=np.array(emb_doc)
y_final=np.array(y)
print(x_final.shape)
print(y_final.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.30, random_state = 30)

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 100)

y_pred = model.predict(x_test)
y_pred = np.rint(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
