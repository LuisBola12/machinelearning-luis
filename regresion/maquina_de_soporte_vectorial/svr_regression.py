# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:29:25 2023

@author: luibv
"""

#SVR

import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
#variable a predecir
y = dataset.iloc[:,2:3].values

#Al tener solo 10 datos, y buscar datos precisos no es bueno dividir el dataset
"""from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state = 0)"""
#escalado de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#Ajustar la regresion con el dataset
#importante, cuando se utiliza SVR, es necesario escalar los datos
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(x,y)


#Prediccion para modelos con SVR
prediction = regression.predict(sc_x.transform([[6.5]]))
prediction = sc_y.inverse_transform(prediction.reshape(-1,1))

#visualizacion de los resultados 


x_grid = npy.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)

inverse_x = sc_x.inverse_transform(x)
invserse_y = sc_y.inverse_transform(y)
inverse_X_grid = sc_x.inverse_transform(x_grid)


pyplot.scatter(inverse_x,invserse_y,color = "red")
pyplot.plot(inverse_X_grid,sc_y.inverse_transform(regression.predict(x_grid).reshape(-1,1)),color = "blue")
pyplot.title("Modelo Regresion SVR")
pyplot.xlabel("Titulo eje X")
pyplot.ylabel("Titulo eje Y")
pyplot.show()


