#Importar librerias 
import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Data.csv')
var_independientes = dataset.iloc[:,:-1].values
#variable a predecir
var_dependiente = dataset.iloc[:,3].values
#codificar datos categoricos
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#categorizar independientes
labelencoder_inde = LabelEncoder()
var_independientes[:,0] =  labelencoder_inde.fit_transform(var_independientes[:,0])
from sklearn.compose import ColumnTransformer
# [0] corresponde a las columnas a ser cambiadas
# remainder indica que el resto no van a ser tocadas
column_transformer = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough')
var_independientes = npy.array(column_transformer.fit_transform(var_independientes),dtype=float)
#categorizar dependiente
labelencoder_depen = LabelEncoder()
var_dependiente = labelencoder_depen.fit_transform(var_dependiente)