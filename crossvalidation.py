from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from   sklearn.linear_model import LinearRegression
import pandas as pd
import pandas as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error



df = pd.read_csv('advertising.csv')

colunas = ['TV', 'newspaper', 'radio']

x = df[colunas].values

y = df['sales'].values


#Validação Cruzada --------- Cross Val Score e KFolds
seed = 1234567
kf = KFold(n_splits=4, shuffle=True, random_state=seed)


i = 0
for train, test in kf.split(x):
    i = i + 1
    print(f'Separação: {i}: \Treino: {x[train]} \n\n Teste: {x[test]}\n')
 
i = 0
for train, test in kf.split(y):
    i = i + 1
    print(f'Separação: {i}: \Treino: {y[train]} \n\n Teste: {y[test]}\n') 
 
regressor = LinearRegression()
regressor.fit(x[train], y[train])
predict= regressor.predict(x[test])

scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_squared_error')
mse_scores=-scores
mean_mse= mse_scores.mean()
print(mse_scores)
print(mean_mse)

X = df[colunas].values
y = df['sales'].values
X2 = sm.add_constant(X) #Adiciona costante ao modelo
est = sm.OLS(y, X2) #Criando um mo delo
est2 = est.fit() #Treinando o modelo estatístico
print(est2.summary()) #Sumário com estatísticas descritivas
print("O modelo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(regressor.intercept_, regressor.coef_[0], regressor.coef_[1], regressor.coef_[2]))