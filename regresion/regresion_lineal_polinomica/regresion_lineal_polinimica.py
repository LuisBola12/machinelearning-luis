

#Regresion Lineal Polinomica


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


# Regresion lineal simple, para comparar resultados
from sklearn.linear_model import LinearRegression

lin_regresion = LinearRegression()
lin_regresion.fit(x,y)
#prediction = lin_regresion.predict(x)

#Ajustar la regresion lineal polinomica con todo el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

poly_lin_reg = LinearRegression()
poly_lin_reg.fit(x_poly,y)
#poly_prediction = poly_lin_reg.predict(x)

#visualizacion de los resultados del modelo lineal
pyplot.scatter(x,y,color = "red")
pyplot.plot(x,lin_regresion.predict(x),color = "blue")
pyplot.title("Modelo Regresion Lineal")
pyplot.xlabel("Posicion del empleado")
pyplot.ylabel("Sueldo en dolares")
pyplot.show()

#visualizacion de los resultados del modelo polinomico
x_grid = npy.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
pyplot.scatter(x,y,color = "red")
pyplot.plot(x_grid,poly_lin_reg.predict(poly_reg.fit_transform(x_grid)),color = "blue")
pyplot.title("Modelo Regresion Polinomica")
pyplot.xlabel("Posicion del empleado")
pyplot.ylabel("Sueldo en dolares")
pyplot.show()

#Prediccion de ambos modelos
print(lin_regresion.predict([[6.5]])) 
print(poly_lin_reg.predict(poly_reg.fit_transform([[6.5]])))