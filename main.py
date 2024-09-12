from flask import Flask,render_template,request,redirect,url_for

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model



app = Flask(__name__)

@app.route('/',methods = ["POST","GET"])
def hello_world():
    if request.method == "POST":
        input1 = request.form["name"]
        if input1:
            return redirect(url_for("hello_user",input = input1))
    return render_template('index.html')

@app.route('/user/<float:input>',methods = ["POST","GET"])
def hello_user(input):
    if request.method == "POST":
        input1 = request.form["name"]
        if input1:
            return redirect(url_for("hello_user",input = input1))
        
    salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
    salary.columns

    y = salary['Salary']
    X = salary[['Experience Years']]

    X = X.to_numpy()
    y = y.to_numpy()

    y = y.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)


    #f(x) = xw
    #tim a,b



    #tạo một ma trận có X.shape[0] hàng và 1 cột có toàn giá trị 1 ̣(hàng dọc)
    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((one, X), axis = 1)

    #2X2
    A = np.dot(Xbar.T, Xbar)

    b = np.dot(Xbar.T, y)
    w = np.dot(np.linalg.pinv(A),b)



    # print("w = ",w)
    w_0=w[0]
    w_1=w[1]


    # x0 = np.linspace(1,11, 2)
    # y0 = w_0 + w_1*x0


    # plt.plot(X.T, y.T, 'ro')     # data 
    # plt.plot(x0, y0)               # the fitting line
    # plt.axis([0, 12, 35000, 125000])
    # plt.xlabel('Height (cm)')
    # plt.ylabel('Weight (kg)')
    # plt.show()

    # while(True):
    years = input
    ketqua = w_1*years + w_0
    #     if(years==0):
    #         break


    # regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    # regr.fit(Xbar, y)


    return f"""<div style = "font-size:50;margin-left:200px;margin-top:200px;">PREDICT SALARY OF PERSON WITH {years} EXPERIENCE YEARS is :  <div style = "color : red;display:inline-block;">{round(float(ketqua))}</div></div>"""


if __name__=="__main__":
    app.run(debug = True)