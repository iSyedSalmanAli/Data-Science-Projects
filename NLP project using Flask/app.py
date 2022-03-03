from flask import Flask, render_template,url_for,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=["POST"])
def predict():
    emails = pd.read_csv("emails.csv", encoding =  'latin-1')
    emails = emails.drop(["Unnamed: 2" ,"Unnamed: 3" ,"Unnamed: 4"],axis = 1)
    emails.columns = ["class","message"]
    emails["label"] = emails["class"].replace("ham" , 0).replace("spam" , 1)
    x = emails["message"]
    y = emails["label"]
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
    naive = MultinomialNB()
    naive.fit(x_train,y_train)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data)
        my_prediction = naive.predict(vect)[0]
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug = True)
