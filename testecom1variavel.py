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
df = df.drop(['Unnamed: 0','radio', 'newspaper'], axis=1)
print(df)
print(df.describe())


#Modelo estatístico e métricas de precisão (3 Var independ.)

#Teste de Multicolinearidade e Dimensionalidade 
#Correlação das variáveis independentes
print(df.corr()) #Multicolinearidade

#Validção Cruzada ----- Tests como forma de analisar a precisão do modelo, quanto maior for o erro, maior a dimensionalidade(Métrica de precisão).
x_train = df.drop(['sales'],axis=1)
y_train = df['sales']

model = LinearRegression() #Dimensionalidade
scores = cross_val_score(model, x_train, y_train, cv=5)
mse_scores = -scores
mean_mse = mse_scores.mean()
print('Média dos erros: ',mean_mse) # mean_mse = 3.07 ----- Deve-se normalizar os dados, realizando feature engineering.
#Variáveis independentes possuem dados de diferentes escalas, isso pode fazer com que o modelo não seja preciso, deve reduzir a dimensionalidade.


#Construção do modelo estatísico 
#Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
X = df['TV']
y = df['sales']
X2 = sm.add_constant(X) #Adiciona costante ao modelo
est = sm.OLS(y, X2) #Criando um mo delo
est2 = est.fit() #Treinando o modelo estatístico
print(est2.summary()) #Sumário com estatísticas descritivas

#Previsão ------- Somente, quando o modelo estiver ajustado e preciso.
#Regressão Linear Múltipla
X = df['TV'].values.reshape(-1,1) #Determinando colunas que serão treinadas
y = df['sales'].values.reshape(-1,1) # Determinando colunas que será prevista

reg = LinearRegression() #Modelo de Regressão Linear
reg.fit(X, y) #Treinamento do modelo
f_multpredict = reg.predict(X) #Prediccao do modelo
f_multpredict = f_multpredict.flatten()
mean_predict = f_multpredict.mean()
print(f_multpredict)
print("O modelo é: Vendas = {:.5} + {:.5}*TV".format(reg.intercept_[0], reg.coef_[0][0]))