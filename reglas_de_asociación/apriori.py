
#Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)

#procesar los datos para el algoritmo de apriori
transactions = []
for index in range(0,7501):
    transactions.append([str(dataset.values[index,index_2]) for index_2 in range(0,20)])
    
#Entrenar algoritmo de apriori

from apyori import apriori

rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2, 
                min_lift = 3 , min_length = 2)
#visualizacion de los resultados
results = list(rules)
print(results[0])
print(results[1])
print(results[2])
print(results[3])