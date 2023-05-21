

#Muestreo Thompson
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')



#Algoritmo de muestreo thompson
N = 10000
ads = 10
number_of_rewards = [0] * ads
number_of_no_rewards = [0] * ads
ads_selected = []
total_reward = 0

for i in range(0,N):
    max_random = 0
    ad = 0
    for j in range(0,ads):
        random_beta = random.betavariate(number_of_rewards[j]+1,number_of_no_rewards[j]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = j
    ads_selected.append(ad)
    reward = dataset.values[i,ad]
    if reward == 1:
        number_of_rewards[ad] = number_of_rewards[ad] + 1
    else:
        number_of_no_rewards[ad] = number_of_no_rewards[ad] + 1
    total_reward = total_reward + reward
    
#Histograma de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del anuncio")
plt.ylabel("Frecuencia de visualizacion del anuncio")
plt.show()