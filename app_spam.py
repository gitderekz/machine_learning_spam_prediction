import os

# import packages
import numpy as np
from flask import Flask,request,jsonify,render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as pk

app = Flask(__name__)
# model = pk.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    # payload = [np.array(int_features)]
    print('Payload: ',int_features)
    phone = [int_features[0]]
    message = [int_features[1]]

    # # *********
    # # import data
    # spam_df = pd.read_csv('spam.csv')  # spam dataframe
    #
    # # # inspect data
    # # spam_df.groupby('Category').describe
    #
    # # create new column spam(vectorize ham/spam to numerical data)
    # spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    #
    # # create train/test split
    # # x = email content(features), y = (label)
    # x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)
    #
    # # find word count in mesage
    # cv = CountVectorizer()
    # # make it to matrix
    # x_train_count = cv.fit_transform(x_train.values)
    # x_train_count.toarray()
    #
    # # train model
    # model = MultinomialNB()
    # model.fit(x_train_count, y_train)
    #
    # # Save model and vectorizer
    # with open('model.pkl', 'wb') as model_file:
    #     pk.dump((model, cv), model_file)

    # Use stored model
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as model_file:
            model, cv = pk.load(model_file)

    # # pre-test
    # valid_email = ['niaje chaliangu?']
    # valid_email_count = cv.transform(valid_email)
    # result = model.predict(valid_email_count)
    # print(result[0])
    # invalid_email = ['congratulations you have won money, click here to receive!']
    # invalid_email_count = cv.transform(invalid_email)
    # result = model.predict(invalid_email_count)
    # print(result[0])

    # # test model
    # x_test_count = cv.transform(x_test)
    # test_result = model.score(x_test_count, y_test)
    # print(test_result)

    # message = invalid_email_count
    # ***********

    message = cv.transform(message)
    print('message: ',message)
    prediction = model.predict(message)
    print('prediction: ',prediction)

    output = prediction[0]
    if output == 1:
        output = 'A SPAM'
    else:
        output = "NOT A SPAM"
    # return render_template('index_spam.html',prediction_text='RESULT = []'.format(output))
    return render_template('index.html',prediction_text='THIS IS {}'.format(output))
    # return {"result":'{}'.format(output)}

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Bind to all interfaces