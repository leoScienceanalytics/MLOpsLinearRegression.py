#Regressão Linear Múltipla, uma variável independente
#Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from   sklearn.linear_model import LinearRegression
from   sklearn.metrics import r2_score
import statsmodels.api as sm



df = pd.read_csv('advertising.csv')
print(df)
print(df.describe())
df.drop(['Unnamed: 0'], axis=1)


plt.figure(figsize = (16,8))
plt.scatter(df['TV'], df['sales'], c='red')
plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()


X = df['TV'].values.reshape(-1,1)
y = df['sales'].values.reshape(-1,1)


reg = LinearRegression()
reg.fit(X, y)


print("O modelo é: Vendas = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))



def calcular_valores(z):
    resultados = []  # Uma lista para armazenar os resultados calculados para cada linha
    forecast_out = 30
    

    for w in z:
        # Realiza o cálculo desejado para cada valor na coluna
        y = 7.0326 + 0.047537*w 
        resultados.append(y)  # Adiciona o resultado à lista de resultados
        valores_desejados = resultados[:forecast_out]  

    return valores_desejados  # Retorna a lista de resultados calculados
    

#Executando o 'def'
tv = df['TV']
resultados_calculados = calcular_valores(tv)


# Os resultados calculados para cada linha da coluna serão armazenados em "resultados_calculados"
print('Predict :' ,resultados_calculados)


plt.figure(figsize = (16,8))
plt.plot(resultados_calculados, c='red')
plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()





