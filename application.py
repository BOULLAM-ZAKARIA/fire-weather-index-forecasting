from flask import Flask,render_template,redirect,request,url_for
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application=Flask(__name__)
app=application

#import pickle files
ridge_model=pickle.load(open("model/ridge.pkl","rb"))
scaler_model=pickle.load(open("model/scaler.pkl","rb"))


@app.route("/") #route for home page
def index(): # create home page
    return render_template("index.html")


@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get("Temperature")) # name text box
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template("home.html",results=result[0])
    else :
        return render_template('home.html')








if __name__=='__main__':
    app.run(host="192.168.1.13",port=5000,debug=True)