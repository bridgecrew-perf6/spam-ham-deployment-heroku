#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template, url_for
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)


# In[ ]:


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


# In[ ]:


if __name__=="__main__":
    app.run(debug=True)

