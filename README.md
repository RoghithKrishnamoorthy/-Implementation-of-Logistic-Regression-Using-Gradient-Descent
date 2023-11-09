# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Predict the values of array.

5. Calculate the accuracy, confusion and classification report by importing the required modules
from sklearn.

8. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Roghith K
RegisterNumber:  212222040135
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="cadetblue")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="plum")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot),color="cadetblue")
plt.show()

def costFunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad


x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="mediumpurple")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted",color="pink")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return(prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)
*/
```

## Output:
![ex 5 1.Array Value of x](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/f79f8d9b-489a-4a75-bf9b-e7318acdc0b3)
![ex 5 2.Array Value of y](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/fe615bf7-648f-48f3-973e-8f8d6907ab59)
![ex 5 3.Exam 1-Score Graph](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/edd347b7-834a-4842-9994-fefd595c4140)
![ex 5 4.Sigmoid function graph](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/14dbdba5-36dc-41a6-807b-9619c87fe4dd)
![ex 5 5.x_train_grad value](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/08db8880-0951-4a15-b748-6b0fb4d9fb66)
![ex 5 6.y_train_grad value](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/f44990f4-4cde-4c95-8854-c3fbbda2cf8c)
![ex 5 7.Print res.x](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/0ac92e23-5ba5-45ee-ad8d-9e05ef4010dd)
![ex 5 8.Decision boundary-graph for exam score](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/c859edb4-bbfe-42e0-99e0-477966f00647)
![ex 5 9.Probability value](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/ba7e24de-e83f-474d-bffa-16691d19023b)
![ex 5 10.Prediction value of mean](https://github.com/RoghithKrishnamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475474/39cf79cc-c641-4315-a34c-18177a5c7a53)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

