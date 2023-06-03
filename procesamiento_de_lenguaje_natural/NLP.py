#Natural language procesing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as npy

#Cargar datos
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting = 3)

#Limpieza del texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for index in range(0,1000):
    #Eliminacion de todo lo que no sean letras
    review = re.sub('[^a-zA-Z]',' ',dataset["Review"][index])
    #Pasar las letras a minusculas
    review = review.lower()
    #Eliminacion de palabras irrelevantes(Articulos, preposiciones, conjucciones, etc)
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Crear bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Dividir el dataset en conjunto entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)

#Ajustar el modelo de clasificador
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

#Predecir los resultados
y_pred = classifier.predict(X_test)

#Matriz de confusion para ver si las predicciones calzan con el test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acierto = (cm[0][0] + cm[1][1]) / (cm[0][1] + cm[1][0] +cm[0][0] + cm[1][1])
precision = (cm[0][0]) / (cm[0][0] + cm[0][1])
recall = (cm[0][0])/(cm[0][0] + cm[1][0])
f1_score = 2*precision*recall/(precision+recall)