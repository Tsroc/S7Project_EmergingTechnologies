#!/usr/bin/env python
# coding: utf-8

# In[1]:


# info here: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# http://www.smartcomputerlab.org/m6/Lab1.regression.tf.keras.pdf


# ## Imports

# In[2]:


# Data frames.
import pandas as pd

# Plotting.
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning.
import keras as kr
import tensorflow as tf

import numpy as np


# ## Class: Linear Regression

# In[3]:


class LinearRegression:
    
    # Constructor, defining # of epochs here.
    def __init__(self, epoch):
        print(f'Linear Regression model to be created with {epoch} epochs.')
        self.epoch = epoch

    # Loads the data from the file, seperates into X and y.
    def LoadData(self, fname):  
        # Limiting to .3 decimal points.
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # Reading fname, skipping headings
        self.dataset = np.loadtxt(fname, delimiter=",", skiprows=1)
        # Windspeed data.
        self.X = self.dataset[:,0]
        # Power data.
        self.y = self.dataset[:,1]
        
    # Creates the model.
    def CreateModel(self):
        # Create model.
        from keras.models import Sequential
        from keras.layers import Dense

        thisShape = 1

        self.model = Sequential()
        self.model.add(Dense(12, input_dim=thisShape, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(thisShape, activation='relu'))

        # Train model.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.tf_history = self.model.fit(self.X, self.y, epochs=self.epoch, verbose=True)
        
        # Prediction.
        self.predictions = self.model.predict(self.X)
        
    def PredictPower(self, windspeed):
        speed = [windspeed]
        prediction = self.model.predict(speed)
        print("Power prediction: %.3f" % prediction)
        return prediction
        
    # Displays the model loss.
    def DisplayLoss(self):
        # displayLoss():
        plt.title("Linear Regression")
        plt.plot(self.tf_history.history['loss'], color="red")
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()
        
    # Displays the model predictions.
    def DisplayPrediction(self):
        # displayPlot():
        plt.figure(figsize=(12,7))
        plt.title('Results')
        plt.scatter(self.X, self.y, label='Data $(X, y)$')
        plt.plot(self.X, self.predictions, color='red', label='Linear Regression',linewidth=3.0)
        plt.xlabel('$X$', fontsize=20)
        plt.ylabel('$y$', fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.show()
    


# ## Class: Polynomial Regression

# In[4]:


class PolynomialRegression:
    
    # Constructor, defining # of epochs and poly features here.
    def __init__(self, epoch, polyDegree):
        print(f'Polynomial Regression model to be created with {polyDegree} Polynominal features, {epoch} epochs.')
        # PolynomialFeatures.
        self.polyDegree = polyDegree
        self.epoch = epoch

    # Loads the data from the file, seperates into X and y. Allows for polynomial features.
    def LoadData(self, fname):  
        # Limiting to .3 decimal points.
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # Reading fname, skipping headings
        self.dataset = np.loadtxt(fname, delimiter=",", skiprows=1)
        # Windspeed data.
        self.X = self.dataset[:,0]
        # Power data.
        self.y = self.dataset[:,1]
        
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(degree=self.polyDegree)
        self.X_2 = self.poly.fit_transform(self.X.reshape(-1,1))
        
    # Creates the model.
    def CreateModel(self):
        # Create model.
        from keras.models import Sequential
        from keras.layers import Dense

        thisShape = 1+self.polyDegree

        self.model = Sequential()
        self.model.add(Dense(64, input_dim=thisShape, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(thisShape, activation='relu'))

        # Train model.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.pr_history = self.model.fit(self.X_2, self.y, epochs=self.epoch, verbose=True)
                
        # Prediction.
        self.predictions = self.model.predict(self.X_2)
        
    # Makes a prediction based on the model.
    def PredictPower(self, windspeed):
        value = np.array([windspeed])
        aValue = self.poly.fit_transform(value.reshape(-1,1))
        prediction = self.model.predict(aValue)
        print("Power prediction: %.3f" % prediction[0][1])
        return prediction[0][1]
        
    # Displays the model loss.
    def DisplayLoss(self):
        plt.title("Polynomial Regression")
        plt.plot(self.pr_history.history['loss'], color="red")
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()
        
    # Displays the model predictions.
    def DisplayPrediction(self):
        plt.figure(figsize=(12,7))
        plt.title('Results')
        plt.scatter(self.X_2[:,1], self.y, label='Data $(X, y)$')
        plt.plot(self.X_2[:,1], self.predictions[:,1], color='orange', label='Polynomial Regression',linewidth=3.0)
        plt.xlabel('$X$', fontsize=20)
        plt.ylabel('$y$', fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.show()
    


# ## Testing
# Testing the LinearRegression and PolynomialRegression models.

# In[5]:


class TestClass:
    fname = "Windspeed.txt"
  
    
    def CreateLinearRegression(self):
        self.lr = LinearRegression(200)
        self.lr.LoadData(self.fname)

    def CreatePolynomialRegression(self):
        self.pr = PolynomialRegression(200, 3)
        self.pr.LoadData(self.fname)   
        
    def TrainLR(self):
        self.lr.CreateModel()
    
    def TrainPR(self):
        self.pr.CreateModel()
        
    def DisplayLoss(self):
        self.lr.DisplayLoss()
        self.pr.DisplayLoss()
        
    def Predictions(self):
        self.lr.PredictPower(20)
        self.pr.PredictPower(20)

    def DisplayPredictions(self):
        self.lr.DisplayPrediction()
        self.pr.DisplayPrediction()
        