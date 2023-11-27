# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as pk

# import data
spam_df = pd.read_csv('spam.csv') #spam dataframe

# # inspect data
# spam_df.groupby('Category').describe

# create new column spam(vectorize ham/spam to numerical data)
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x=='spam' else 0)

# create train/test split
# x = email content(features), y = (label)
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message,spam_df.spam,test_size = 0.25)

# find word count in mesage
cv = CountVectorizer()
# make it to matrix
x_train_count = cv.fit_transform(x_train.values)
x_train_count.toarray()

# train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# store model
pk.dump(model,open('model.pkl','wb'))

# use stored model
model = pk.load(open('model.pkl','rb'))

# pre-test
valid_email = ['niaje chaliangu?']
valid_email_count = cv.transform(valid_email)
result = model.predict(valid_email_count)
print(result)
invalid_email = ['congratulations you have won money, click here to receive!']
invalid_email_count = cv.transform(invalid_email)
result = model.predict(invalid_email_count)
print(result)

# test model
x_test_count = cv.transform(x_test)
test_result = model.score(x_test_count, y_test)
print(test_result)
