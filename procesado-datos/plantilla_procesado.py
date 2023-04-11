

#Plantilla de Pre Procesado

#Importar librerias 
import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Data.csv')

var_independientes = dataset.iloc[:,:-1].values

#variable a predecir
var_dependiente = dataset.iloc[:,3].values

#Dividir el dataset en conjunto entrenamiento y testing
from sklearn.model_selection import train_test_split
var_in_train,var_in_test,var_depen_train,var_depen_test = train_test_split(var_independientes,var_dependiente,test_size = 0.2,random_state = 0)


#escalado de variables
"""from sklearn.preprocessing import StandardScaler
std_scl_in = StandardScaler()
var_in_train = std_scl_in.fit_transform(var_in_train)
var_in_test = std_scl_in.transform(var_in_test)"""

