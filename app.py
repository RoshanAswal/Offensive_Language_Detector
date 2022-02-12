from crypt import methods
from flask import Flask, render_template,request
import numpy as np
import pickle 

app=Flask(__name__)

model=pickle.load(open("model1.pkl",'rb'))
cv=pickle.load(open("cv1.pkl",'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])
def check():
    instance=str(request.form['text_box'])
    vec=cv.transform([instance])
    prediction=model.predict(vec)
    result=""
    if prediction[0]==1:
        result="Offensive"
    else:
        result="Non-Offensive"
    return render_template("index.html",answer=result)

if __name__=='__main__':
    app.run(debug=True)
