#Importar librerias 
import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Data.csv')

var_independientes = dataset.iloc[:,:-1].values

#variable a predecir
var_dependiente = dataset.iloc[:,3].values

#tratamiento de los NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = npy.nan, strategy = "mean")
imputer = imputer.fit(var_independientes[:,1:3])
var_independientes[:,1:3] = imputer.transform(var_independientes[:,1:3])