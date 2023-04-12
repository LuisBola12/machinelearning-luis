
# Regresion Lineal Multiple

#Plantilla de Pre Procesado

#Importar librerias 
import numpy as npy
import matplotlib.pyplot as pyplot
import pandas as pds 

#importar el dataset
dataset = pds.read_csv('50_Startups.csv')

x = dataset.iloc[:,:-1].values

#variable a predecir
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#categorizar independientes
labelencoder_inde = LabelEncoder()
x[:,3] =  labelencoder_inde.fit_transform(x[:,3])
from sklearn.compose import ColumnTransformer
# [3] corresponde a las columnas a ser cambiadas
# remainder indica que el resto no van a ser tocadas
column_transformer = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough')
x = npy.array(column_transformer.fit_transform(x),dtype=float)
# Evitar la trampa de las variables dumy (me quedo con una menos del total)
x = x[:,1:]

#Dividir el dataset en conjunto entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


#escalado de variables
"""from sklearn.preprocessing import StandardScaler
std_scl_in = StandardScaler()
var_in_train = std_scl_in.fit_transform(var_in_train)
var_in_test = std_scl_in.transform(var_in_test)"""


# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

#prediccion de los resultados en el conjunto de testing
y_prediction = regression.predict(x_test)


#Contruir el modelo optimo de RLM utilizando la eliminacion hacia atras 
import statsmodels.api as sm
# agregar una columna de 1 que corresponde al coeficiente del termino independiente 
x = npy.append(arr = npy.ones((50,1)).astype(int), values = x, axis = 1)

# variable que guarda el numero optimo de variables independientes
x_opt = x[:, [0,1,2,3,4,5]]
SL = 0.05
regression_OLS = sm.OLS(endog=y,exog=x_opt.tolist()).fit()
print(regression_OLS.summary())
#eliminacion de segunda variable dumy por alto valor p 0.990
x_opt = x[:, [0,1,3,4,5]]
SL = 0.05
regression_OLS = sm.OLS(endog=y,exog=x_opt.tolist()).fit()
print(regression_OLS.summary())

#eliminacion de tercera variable dumy por alto valor p 0.940
x_opt = x[:, [0,3,4,5]]
SL = 0.05
regression_OLS = sm.OLS(endog=y,exog=x_opt.tolist()).fit()
print(regression_OLS.summary())

#eliminacion de variable gastos administartivos por alto valor p 0.602
x_opt = x[:, [0,3,5]]
SL = 0.05
regression_OLS = sm.OLS(endog=y,exog=x_opt.tolist()).fit()
print(regression_OLS.summary())

# Se elimina la variable de gastos de marketing por superior al valor p 0.060
# se hace por ser estrictos, mas no es lo correcto, ya que lo estamos convirtiendo
# en un modelo de regresion simple
x_opt = x[:, [0,3]]
SL = 0.05
regression_OLS = sm.OLS(endog=y,exog=x_opt.tolist()).fit()
print(regression_OLS.summary())


#creacion de un metodo que realiza la eliminacion hacia atras automatica
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = npy.delete(x, j, 1)    
        print(regressor_OLS.summary())    
    return x 
func_x_opt = x[:, [0, 1, 2, 3, 4, 5]]
modelo_x = backwardElimination(func_x_opt, SL)