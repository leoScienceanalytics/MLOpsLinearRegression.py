#Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from   sklearn.linear_model import LinearRegression
from   sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #Função de treino e teste, importada do SKlearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#Conectando base de dados
df = pd.read_csv('advertising.csv')
df = df.drop(['Unnamed: 0'], axis=1)
print(df)
print(df.describe())

#Teste de Multicolinearidade e Dimensionalidade 
#Correlação das variáveis independentes
print(df.corr()) #Multicolinearidade

#Validção Cruzada ----- Tests como forma de analisar a precisão do modelo, quanto maior for o erro, maior a dimensionalidade(Métrica de precisão).
x_train = df.drop(['sales'], axis=1)
y_train = df['sales']

model = LinearRegression() #Dimensionalidade
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
mean_mse = mse_scores.mean()
print('Média dos erros: ',mean_mse) # mean_mse = 3.07 ----- Deve-se normalizar os dados, realizando feature engineering.
#Variáveis independentes possuem dados de diferentes escalas, isso pode fazer com que o modelo não seja preciso, deve reduzir a dimensionalidade.

#Feature engineering













plt.figure(figsize = (10,5))
plt.scatter(df['TV'], df['sales'], c='red')
plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()

#Construção do modelo de Regressão Linear
#É necessário usar p .values.reshape para que possa ser definida a equação do modelo de regressão linear.
X = df['TV'].values.reshape(-1,1)
y = df['sales'].values.reshape(-1,1)

#Regressão Linear Simples
reg = LinearRegression()
reg.fit(X, y)
f_predict = reg.predict(X) #Utilizou o método Predict paa que as previsões fossem feitas

print("O modelo de Regressão --> Sales = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0])) #Função do primeiro grau
print('Predict:',f_predict) 


plt.figure(figsize = (10,5))
plt.plot(f_predict, c='red')
plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()


plt.figure(figsize = (10,5))
plt.scatter(df['TV'], df['sales'], c='red')


plt.plot( df['TV'], f_predict, c='blue', linewidth=3, linestyle=':')

plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show() #Função do primeiro grau

#Precisão do modelo
X = df['TV']
y = df['sales']
X2 = sm.add_constant(X)#Adiciona constante ao modeo
est = sm.OLS(y, X2) #Modelo estátístico, método usado para modelar relação entre variáveis
est2 = est.fit() #Treinando o modelo estátistico
print(est2.summary()) #Sumário com estatísticas descritivas

#R² = 0.612, signfica que 61.20% da variação das Vendas é explicada pela variação da TV
#P-Value próximo a 0, pode-se rejeitar a hipótese nula



#Regressão Linear Múltipla
Xs = df.drop(['sales'], axis=1) #Determinando colunas que serão treinadas
y = df['sales'].values.reshape(-1,1) # Determinando colunas que será prevista

reg = LinearRegression() #Modelo de Regressão Linear
reg.fit(Xs, y) #Treinamento do modelo
f_multpredict = reg.predict(Xs) #Prediccao do modelo

print("O modeleo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))
print('Previsão da Regressão Múltipla:', f_multpredict)

plt.figure(figsize = (10,5))
plt.plot(f_multpredict, c='red')
plt.ylabel(" ($) Vendas")
plt.show()

#Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
X = np.column_stack((df['TV'], df['radio'], df['newspaper']))
y = df['sales']

#Precisão do modelo
X2 = sm.add_constant(X) #Adiciona costante ao modelo
est = sm.OLS(y, X2) #Criando um modelo
est2 = est.fit() #Treinando o modelo estatístico
print(est2.summary()) #Sumário com estatísticas descritivas



#Aumentando a precisão do modelo

Xs = df.drop(['newspaper','sales'], axis=1) #Determinando colunas que serão treinadas
y = df['sales'].values.reshape(-1,1) # Determinando colunas que será prevista

reg = LinearRegression() #Modelo de Regressão Linear
reg.fit(Xs, y) #Treinamento do modelo
f_multpredict = reg.predict(Xs) #Prediccao do modelo

print("O modeleo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]))
print('Previsão da Regressão Múltipla:', f_multpredict)

plt.figure(figsize = (10,5))
plt.plot(f_multpredict, c='red')
plt.ylabel(" ($) Vendas")
plt.show()

#Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
X = np.column_stack((df['TV'], df['radio']))
y = df['sales']

#Precisão do modelo
X2 = sm.add_constant(X) #Adiciona costante ao modelo
est = sm.OLS(y, X2) #Criando um modelo
est2 = est.fit() #Treinando o modelo estatístico
print(est2.summary()) #Sumário com estatísticas descritivas