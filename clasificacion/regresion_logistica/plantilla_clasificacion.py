#Plantilla de Clasificación
#Regresion Logistica
#Importar librerias 
import numpy as npy
import matplotlib.pyplot as plt
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,[2,3]].values

#variable a predecir
y = dataset.iloc[:,4].values

#Dividir el dataset en conjunto entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


"""from sklearn.preprocessing import StandardScaler
std_scl_in = StandardScaler()
X_train = std_scl_in.fit_transform(X_train)
X_test = std_scl_in.transform(X_test)"""

#Ajustar el modelo de clasificador
classifier = 
classifier.fit(X_train,y_train)

#Predecir los resultados
y_pred = classifier.predict(X_test)

#Matriz de confusion para ver si las predicciones calzan con el test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Representacion grafica conjunto entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = npy.meshgrid(npy.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     npy.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(npy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(npy.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

#Representacion grafica conjunto test
X_set, y_set = X_test, y_test
X1, X2 = npy.meshgrid(npy.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     npy.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(npy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(npy.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()