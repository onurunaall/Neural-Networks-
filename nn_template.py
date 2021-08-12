# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
The function sogmoid(z) takes as an input both scalar values and vectors.
So if the input is scalar then in computes sigmoid function only to that
scalar and returns the result. If the input is a vector, it performs elementwise
computation of sigmoid function on each element of the vector and returns a vector.
 The same is true for sigmoid_derivative function.
"""
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 - z)

class NeuralNetwork:
    import random
    def __init__(self, inSize, sl2,clsSize, lrt): 
        # Constructor expects:
        # inSize- input size, number of features
        # sl2 - number of neurons in the hidden layer
        # clsSize - number of classes, equals number of neurons in output layer
        # lrt - learning rate
    
        self.iSz=inSize
        self.oSz=clsSize
        self.hSz=sl2
        # Initial assignment of weights 
        np.random.seed(42) ## assigning seed so it generates the same random number all the time. Just to fix the result.
        self.weights1 = (np.random.rand(self.hSz,self.iSz+1)-0.5)/np.sqrt(self.iSz) 
        self.weights2 = (np.random.rand(self.oSz,self.hSz+1)-0.5)/np.sqrt(self.hSz) 
        
        self.output = np.zeros(clsSize)
        
        self.layer1 = np.zeros(self.hSz)
        self.eta = lrt
        
        self.delta1 = 0
        self.delta2 = 0
                
        # this function send forward single sample
    def feedforward(self, x):
        X_biased = np.r_[1,x] # adding the bias term
        z1 = np.dot(self.weights1, X_biased)
        self.layer1 = sigmoid(z1) # alpha2
        
        layer1_biased = np.r_[1,self.layer1] # adding the bias term to the 2nd layer of the NN
        z2 = np.dot(self.weights2, layer1_biased)
        self.output = sigmoid(z2) # alpha3
        
        # this function backpropagates errors of single sample
    def backprop(self, x, trg):
        # the error func. at the output
        sigma3 = trg - self.output 
        # ensuring the dimension of the sigma3 is 3x1
        sigma3 = np.reshape(sigma3, (self.oSz, 1)) 
        
        # creating layer1_biased again to calculate g'(z(2))
        layer1_biased = np.r_[1, self.layer1] 
        # calculating g'(z(2))
        s_deriv2 = sigmoid_derivative(layer1_biased) 
        s_deriv2 = np.reshape(s_deriv2, (self.hSz + 1, 1)) 
        # calculation of sigma2
        sigma2 = np.multiply(np.dot(self.weights2.T, sigma3), s_deriv2)
        
        X_biased = np.r_[1,x] # adding the bias term
        #calculation of deltas
        delta2 = sigma3 * self.layer1
        delta1 = sigma2 * x
        return delta1, delta2
    
    # This function is called for training the data
    def fit(self,X,y,iterNo):
        m=np.shape(X)[0]
        for i in range(iterNo):
            D1=np.zeros(np.shape(self.weights1))
            D2=np.zeros(np.shape(self.weights2))
            #new_error=0
            for j in range(m):
                self.feedforward(X[j])
                yt=np.zeros(self.oSz)
                yt[int(y[j])]=1                 # the output is converted to vector, so if class of a sample is 1, then yt=[0 1 0]
                [delta1,delta2]= self.backprop(X[j],yt)
                D1=D1+delta1
                D2=D2+delta2
            self.weights1= self.weights1+self.eta*(D1/m) # weights1 are updated only ones after one epoch
            self.weights2=self.weights2+self.eta*(D2/m)  # weights2 are updated only ones after one epoch
          
    # This function is called for prediction
    def predict(self,X):
        m=np.shape(X)[0]
        y_proba=np.zeros((m,3))
        y=np.zeros(m)
        for i in range(m):
            self.feedforward(X[i])
            y_proba[i,:]=self.output   # the outputs of the network are the probabilities
            y[i]=np.argmax(self.output) # here we convert the probabilities to classes
        return y, y_proba

# reading the iris data set
iris = datasets.load_iris()
(data, label) = (iris.data, iris.target)
    
# representing the labels as vectors
y = np.zeros((150, 3))
for i in range(150):
    y[i][iris.target[i]] = 1

label = y

num_list = list(range(np.shape(iris.data)[0]))
num_array = np.array(num_list)
# Shuffling the data
np.random.shuffle(num_array)

#splitting the data
for i in range(100):
    X_train = iris.data[num_array[i]].tolist()
    y_train = y[num_array[i]]
    
for i in range(100, 125):
    X_val = iris.data[num_array[i]].tolist()
    y_val = y[num_array[i]]
    
for i in range(125, 150):
    X_test = iris.data[num_array[i]].tolist()
    y_test = y[num_array[i]]    
    
    
# Shuffling the data
#np.take(data, np.random.permutation(data.shape[0]), axis=0, out=data)
# Splitting Data
#from sklearn.model_selection import train_test_split
# splitting the data into two parts; 125 train and 25 test
#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=1/6, train_size=5/6)
# splitting the training data into two parts again; 100 train and 25 validation
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, train_size=0.8)

# parameters
input_size = 4 
hidden_size = 2
class_size = 3
learning_rate = 0.1
epochs = 1000
error_list = []
model_list = []

for i in range(10):
    model = NeuralNetwork(input_size, hidden_size, class_size, learning_rate) 
    model.fit(X_train, y_train, epochs)
    model_list.append(model)
    
for i, model in enumerate(model_list):
    y_predicted = model.predict(X_val) 
    error = mean_squared_error(y_val, y_predicted)
    error_list.append(error)
    min_error = min(error_list)
    if(min_error >= error):
        min_error = error
        min_err_index = i
    else:
        continue
    print("Min Error: ", min_error, "Index: ", i)
    
# calculating the predictions and error of test set
y_predicted = model_list[min_err_index].predict(X_test)
error = mean_squared_error(y_test, y_predicted)

# Printing accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_train, y_predicted))
