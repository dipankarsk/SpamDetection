#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 02:06:12 2018

@author: dipankar
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv",encoding='latin-1')
data=data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data=data.rename(columns={"v1":"label","v2":"text"})
data.tail()

data['label_num'] = data.label.map({'ham':0, 'spam':1})

X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)

#Check_test=["text Sure, whenever you show the fuck up &gt;:("]

vect = CountVectorizer()

vect.fit(X_train)#bag of words

#print(vect.vocabulary_)

X_train_df = vect.transform(X_train)#creates the vector array for train data

X_test_df=vect.transform(X_test)#creates the vector array for test data


#Check_test_df=vect.transform(Check_test)

prediction = dict()

model = MultinomialNB()
model.fit(X_train_df,y_train)
prediction["Multinomial"] = model.predict(X_test_df)

#print(prediction["Multinomial"])

print(accuracy_score(y_test,prediction["Multinomial"]))
