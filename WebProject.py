from flask import Flask, render_template, request
import pickle
import requests
import numpy as np

import pandas as pd
model = pickle.load(open("reg.pkl", "rb"))

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('BA.html')

@app.route("/predict/", methods = ["POST"])
def submit():
    gre_s=int(request.form.get("grescore"))
    toefl_s=request.form.get("toeflscore")
    sop_s=request.form.get("SOP")
    lor_s=request.form.get("LOR")
    cgpa=request.form.get("CGPA")
    r_s=request.form["opt"]
    if(r_s=="Yes"):
        r_s=1
    else:
        r_s=0
    u_r=list()
    prob=list()
    for i in range(1,6):
        values=[[gre_s,toefl_s,i,sop_s,lor_s,cgpa,r_s]]
        predictions=model.predict(values)
        u_r.append(i)
        prob.append(predictions)

    a=pd.Series(u_r,name="University_Rating")
    b=pd.Series(prob,name="Probability_of_Admission")
    a=pd.DataFrame(a)
    res=a.join(b)  
    
    return render_template('table.html',  tables=[res.to_html(classes='data')], titles=res.columns.values)
if __name__ == '__main__':
    app.run(debug=True)