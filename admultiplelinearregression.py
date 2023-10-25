#Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from   sklearn.linear_model import LinearRegression
from   sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error



#Conectando base de dados
df = pd.read_csv('advertising.csv')
df = df.drop(['Unnamed: 0'], axis=1)
print(df)
print(df.describe())

#DataFrame Variáveis independentes
df_nosales = df.drop(['sales'], axis=1)
print('DataFrame var indpend: ',df_nosales)

#Modelo estatístico e métricas de precisão (3 Var independ.)
#Teste de Multicolinearidade e Dimensionalidade 
#Correlação das variáveis independentes
print(df_nosales.corr()) #Multicolinearidade -------- Não há multicolinearidade.


#CrossValidation ----- Tests como forma de analisar a precisão do modelo, quanto maior for o erro, maior a dimensionalidade(Métrica de precisão).
x_train = df_nosales
y_train = df['sales']


num_iterations = 4
quartil_size = len(x_train) // 4 # = 50
partitions = [x_train[i:i+quartil_size] for i in range(0, len(x_train), quartil_size)] #Cria uma lista que contém todos os Quartis, dividos por 4 (25%) ---- X
target_predictions = [y_train[i:i+quartil_size] for i in range(0, len(y_train), quartil_size)] #Cria uma lista que contém todos os Quartis, divididos por 4 (25%) ---- Y 

mse_scoretrain ,mse_scoretest = [], [] #Cria dicionários vazios

for i in range(num_iterations): #Cria um laço que se repete 4 vezes.
    
    x_test, y_test = partitions[i], target_predictions[i] #Chama X(partitions[i]) e Y(target_predictions[1]) de variáveis de teste.
    x_train = np.vstack(partitions[:i] + partitions[i+1:])
    y_train = np.concatenate(target_predictions[:i] + target_predictions[i+1:])
    
    
    modelo = LinearRegression()
    modelo.fit(x_train, y_train)
    y_predtrain = modelo.predict(x_train) #Previsão do treino
    y_predtest = modelo.predict(x_test) #Previsão do Teste 
    y_pred = pd.DataFrame(y_pred)#Previsão do Modelo  
 
    
    
    #Erros Quadráticos Médios
    msetrain = mean_squared_error(y_train, y_predtrain) #Erro Quadrático médio do Treino
    msetest = mean_squared_error(y_test, y_predtest) #Erro Quadrático médio do Teste
    mse_scoretrain.append(msetrain)
    mse_scoretest.append(msetest)
    #Variância deu um Valor baixo ----- 0.21
    
    
    #Construção do modelo estatísico 
    #Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
    X = x_train
    y = y_train
    X2 = sm.add_constant(X) #Adiciona costante ao modelo
    est = sm.OLS(y, X2) #Criando um mo delo
    est2 = est.fit() #Treinando o modelo estatístico
    print(est2.summary()) #Sumário com estatísticas descritivas
    print("O modelo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(modelo.intercept_, modelo.coef_[0], modelo.coef_[1], modelo.coef_[2]))

mse_scoretrain = pd.DataFrame(mse_scoretrain)
mse_scorestest = pd.DataFrame(mse_scoretest) 
print('Erros Quadrados de Treino: ', mse_scoretrain)
print('Erros Quadrados de Teste: ', mse_scoretest)
mean_msetrain = np.mean(mse_scoretrain)
mean_msetest = np.mean(mse_scorestest)
print('Média dos Erros Quadrados de Treino: ', mean_msetrain)
print('Média dos Erros Quadrados de Teste: ', mean_msetest)
#Possuem Baixa Variância, além disso, Erros(Bias) baixo nos dois modelo. Logo, o modelo é ótimo.

print('Previsão de Vendas: ', y_pred)

plt.figure(figsize = (10,5))
plt.plot(y_pred, c='orange')
plt.ylabel('Vendas (em Milhões de US$)')
plt.show()
