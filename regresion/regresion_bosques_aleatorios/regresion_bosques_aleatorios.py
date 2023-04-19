# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:29:25 2023

@author: luibv
"""

# Regresion bosques aleatorios


#Plantilla de Pre Procesado
#Importar librerias 
import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
#variable a predecir
y = dataset.iloc[:,2].values

#Al tener solo 10 datos, y buscar datos precisos no es bueno dividir el dataset
"""from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state = 0)"""
#escalado de variables
"""from sklearn.preprocessing import StandardScaler
std_scl_in = StandardScaler()
var_in_train = std_scl_in.fit_transform(var_in_train)
var_in_test = std_scl_in.transform(var_in_test)"""

#Ajustar la regresion con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators=300,random_state=0)
regression.fit(x, y)
#Prediccion con bosques aleatorios
predictions = regression.predict([[6.5]])


#visualizacion de los resultados 
x_grid = npy.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
pyplot.scatter(x,y,color = "red")
pyplot.plot(x_grid,regression.predict(x_grid),color = "blue")
pyplot.title("Modelo Regresion bosques aleatorios")
pyplot.xlabel("Titulo eje X")
pyplot.ylabel("Titulo eje Y")
pyplot.show()