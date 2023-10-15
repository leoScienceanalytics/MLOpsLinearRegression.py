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


plt.figure(figsize = (10,5))
plt.scatter(df['TV'], df['sales'], c='red')
plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()


X = df['TV'].values.reshape(-1,1)
y = df['sales'].values.reshape(-1,1)


reg = LinearRegression()
reg.fit(X, y)


print("O modelo de Regressão --> Sales = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

f_predict = reg.predict(X) #Utilizou o método Predict paa que as previsões fossem feitas
print('Predict:',f_predict) 



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


plt.figure(figsize = (10,5))
plt.plot(resultados_calculados, c='red')
plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()


plt.figure(figsize = (10,5))
plt.scatter(
    df['TV'], 
    df['sales'], 
    c='red')


plt.plot(
    df['TV'],
    f_predict,
    c='blue',
    linewidth=3,
    linestyle=':'
)

plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()

#Precisão do modelo
X = df['TV']
y = df['sales']
X2 = sm.add_constant(X)#Adiciona constante ao modeo
est = sm.OLS(y, X2) #Modelo estátístico, método usado para modelar relação entre variáveis
est2 = est.fit() #Treinando o modelo estátistico
print(est2.summary()) #Sumário com estatísticas descritivas

#R² = 0.612, signfica que 61.20% da variação das Vendas é explicada pela variação da TV
#P-Value próximo a 0, pode-se rejeitar a hipótese nula

#Regressão Múltipla
Xs = df.drop(['sales', 'Unnamed: 0'], axis=1)
y = df['sales'].values.reshape(-1,1)


reg = LinearRegression()
reg.fit(Xs, y)
f_multpredict = reg.predict(Xs)
print('Previsão da Regressão Múltipla:', f_multpredict)

print("O modleo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))


X = np.column_stack((df['TV'], df['radio'], df['newspaper']))
y = df['sales']

#Precisão do modelo
X2 = sm.add_constant(X) #Adiciona costante ao modelo
est = sm.OLS(y, X2) #Criando um modelo
est2 = est.fit() #Treinando o modelo estatístico
print(est2.summary()) #Sumário com estatísticas descritivas
