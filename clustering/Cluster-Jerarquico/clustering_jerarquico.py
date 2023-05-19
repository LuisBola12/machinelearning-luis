
#Clustering Jerarquico


#Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Obtener el conjunto de datos
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

#Obtener el numero optimo de clusters
#Esto a través de los dendogramas
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show()


#Ajustar el clustering jerarquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


#Visualizacion de los clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 50, c = "red", label = "Tacaños")
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 50, c = "blue", label = "Estandar")
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 50, c = "yellow", label = "Objetivo")
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 50, c = "green", label = "Descuidados")
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 50, c = "orange", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales en miles de $")
plt.ylabel("Puntuacion de gastos (1-100)")
plt.legend()
plt.show()