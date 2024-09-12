import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model

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



print("w = ",w)
w_0=w[0]
w_1=w[1]


x0 = np.linspace(1,11, 2)
y0 = w_0 + w_1*x0


plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([0, 12, 35000, 125000])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

# while(True):
#     years = float(input())
#     print(w_1*years + w_0)
#     if(years==0):
#         break


regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w)




# [[ 0.12093915 -0.01861992]
#  [-0.01861992  0.00361376]]


# [ 2989745.  18040043.3]

# [25673.01576053  9523.65050742]













# plt.plot(X, y, 'ro')
# plt.axis([0,11, 35000, 130000])
# plt.xlabel('Experience Years')
# plt.ylabel('Salary')
# plt.show()











