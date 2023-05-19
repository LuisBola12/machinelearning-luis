
# K-Means
#Importar las librerias de trabajo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values


#Metodo del codo para averiguar el numero optimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++",max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

#Aplicar el metodo de k-means para segmentar el dataset
kmeans = KMeans(n_clusters=5,init = "k-means++",max_iter = 300,n_init = 10,random_state = 0)
y_kmeans = kmeans.fit_predict(X)


#Visualizacion de los clusters 
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 50, c = "red", label = "Precabidos")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 50, c = "blue", label = "Estandar")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 50, c = "yellow", label = "Objetivo")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 50, c = "green", label = "Descuidados")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s = 50, c = "orange", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 100, c = "magenta", label = "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales en miles de $")
plt.ylabel("Puntuacion de gastos (1-100)")
plt.legend()
plt.show()
