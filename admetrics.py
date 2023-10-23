#Código com foco em métricas de precisão


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import statsmodels.api as sm
from   sklearn.metrics import r2_score


df = pd.read_csv('advertising.csv')


#Determinando variáveis-chave
independ_var = ['TV', 'radio', 'newspaper'] 
depend_var = 'sales'
test_size = 0.3

def crossvalidation(df, independ_var, depend_var, test_size):
    x = df[independ_var] 
    y = df[depend_var]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0) 
    #Separando variáveis para treinamento e testes.
    #Validação cruzada
   
    response = [x_train, x_test, y_train, y_test]
    #Vai criar um dicionário onde o cada variável de treino e teste existirá
    
    return response
    #Usada para retorna as responses de treino e teste

x_train, x_test, y_train, y_test  = crossvalidation(df,independ_var,depend_var,test_size); 

# Treinar um modelo de regressão linear
model = LinearRegression()
model.fit(x_train, y_train)

#Precisão do Teste
score = model.score(x_test, y_test)
score = round(score, 2) *100

# Fazer previsões no conjunto de teste
y_pred = model.predict(x_test)

# Calcular o erro total (Erro Quadrático Médio)
mse = mean_squared_error(y_test, y_pred)


# Calcular o viés e a variância
bias = np.mean(y_test - y_pred)
variance = np.mean((y_pred - y_pred.mean()) ** 2)

print(f'Precisão: {score} %')
print(f"Erro Quadrático Médio: {mse}")
print(f"Bias: {bias}")
print(f"Variância: {variance}")

#Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
X = df.drop(['Unnamed: 0','sales'], axis=1)
y = df['sales']
X2 = sm.add_constant(X) #Adiciona costante ao modelo
est = sm.OLS(y, X2) #Criando um mo delo
est2 = est.fit() #Treinando o modelo estatístico
print(est2.summary()) #Sumário com estatísticas descritivas