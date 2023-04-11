#Regresion Lineal Simple


#Plantilla de Pre Procesado

#Importar librerias 
import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Salary_Data.csv')

var_independientes = dataset.iloc[:,:-1].values

#variable a predecir
var_dependiente = dataset.iloc[:,1].values

#Dividir el dataset en conjunto entrenamiento y testing
from sklearn.model_selection import train_test_split
var_in_train,var_in_test,var_depen_train,var_depen_test = train_test_split(var_independientes,var_dependiente,test_size = 1/2,random_state = 0)


#escalado de variables 
# En regresion simple no es necesario
"""from sklearn.preprocessing import StandardScaler
std_scl_in = StandardScaler()
var_in_train = std_scl_in.fit_transform(var_in_train)
var_in_test = std_scl_in.transform(var_in_test)"""

#convertir variables a un array2D
var_in_train = npy.array(var_in_train).reshape(-1,1)
var_depen_train = npy.array(var_depen_train).reshape(-1,1)
var_in_test = npy.array(var_in_test).reshape(-1,1)
var_depen_test = npy.array(var_depen_test).reshape(-1,1)
# Crear modelo de regresion lineal simple con
# el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regresion = LinearRegression()
regresion.fit(var_in_train,var_depen_train) 


#Predecir el conjunto de test
#predict pide que el train y el test tengan la misma cantidad de datos
depen_prediccion = regresion.predict(var_in_test)

#Visualizacion de los resultados de entrenamiento
pyplot.scatter(var_in_train,var_depen_train, color = "red")
pyplot.plot(var_in_train,regresion.predict(var_in_train),color = "blue")
pyplot.title("Sueldo vs Anos de experiencia (Conjunto de entrenamiento)")
pyplot.xlabel("Años de experiencia")
pyplot.ylabel("Sueldo en dolares")
pyplot.show()
# Visualizacion de los datos de test
pyplot.scatter(var_in_test,var_depen_test, color = "red")
pyplot.plot(var_in_train,regresion.predict(var_in_train),color = "blue")
pyplot.title("Sueldo vs Anos de experiencia (Conjunto de testing)")
pyplot.xlabel("Años de experiencia")
pyplot.ylabel("Sueldo en dolares")
pyplot.show()