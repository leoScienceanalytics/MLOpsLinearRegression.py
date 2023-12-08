import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error



class leitorDados:
    def __init__(self, arquivo): #Iniciar o construtor
        self.arquivo = arquivo
        self.dados = None
        self.partitions = []
        self.target_partitions = []
                    
    def ler_dados(self): #Ler os dados
        try:
            self.dados = pd.read_csv(self.arquivo)
            print("Dados lidos com sucesso!")
        except FileNotFoundError:
            print(f"Erro: O arquivo '{self.arquivo}' não foi encontrado.")
        except Exception as e:
            print(f"Erro ao ler os dados: {str(e)}")
            
    def exibir_dados(self): #Plotar os dados
        if self.dados is not None:
            print(self.dados)
        else:
            print("Por favor, leia os dados primeiro usando o método 'ler_dados'.")

    def tratar_dados(self): #Realizar o tratamento
        self.dados = self.dados.drop(['Unnamed: 0'], axis=1)
        print(self.dados.describe())
        self.df_nosales = self.dados.drop(['newspaper','sales'], axis=1)
        print(self.df_nosales)
        print(self.dados.columns) #Mostrar as colunas do dataframe
         
    def separar_dados(self): #Engenharia de features
        #Conjunto de Treino - Features
        self.x_train = self.df_nosales
        print('Conjunto de Features:')
        print(self.x_train)
        
        #Conjunto de Treino - Target
        self.y_train = self.dados['sales']
        print('Conjunto de Target:')
        print(self.y_train)
        
    def cross_validation(self): #Validação Cruzada
        num_iterations = 4
        self.quartil_size = len(self.x_train)// num_iterations # =50
        self.partitions = [self.x_train[i:i+self.quartil_size] for i in range(0, len(self.x_train), self.quartil_size)]
        self.target_partitions = [self.y_train[i:i+self.quartil_size] for i in range(0, len(self.y_train), self.quartil_size)]
        print('Tamanho do Quartil: ',self.quartil_size)
        
        self.mse_scoretrain, self.mse_scoretest = [], []
        
        for i in range(self.quartil_size):
            
            self.x_test, self.y_test = self.partitions[i], self.target_partitions[i] #Chama X(partitions[i]) e Y(target_predictions[1]) de variáveis de teste.
            self.x_train = np.vstack(self.partitions[:i] + self.partitions[i+1:])
            self.y_train = np.concatenate(self.target_partitions[:i] + self.target_partitions[i+1:]) #Validação vai até aqui.
            print('Conjunto X teste:')
            print(self.x_test)
        
            #Construção do modelo de regressão linear
            modelo = LinearRegression()
            modelo.fit(self.x_train, self.y_train)
            self.y_predtrain = modelo.predict(self.x_train) #Previsão do treino
            self.y_pred = modelo.predict(self.x_test) #Previsão do Teste 
            
            
            #Erros Quadráticos Médios
            self.msetrain = mean_squared_error(self.y_train, self.y_predtrain) #Erro Quadrático médio do Treino
            self.msetest = mean_squared_error(self.y_test, self.y_pred) #Erro Quadrático médio do Teste
            self.mse_scoretrain.append(self.msetrain)
            self.mse_scoretest.append(self.msetest)
     
            #Modelo de Regressão pelo Métodos OLS ----- Usado para medir precisão do Modelo de Regressão Linear Múltipla.
            X = self.x_train
            y = self.y_train
            X2 = sm.add_constant(X) #Adiciona costante ao modelo
            est = sm.OLS(y, X2) #Criando um mo delo
            est2 = est.fit() #Treinando o modelo estatístico
            print(est2.summary()) #Sumário com estatísticas descritivas
            print("O modelo é: Vendas = {:.5} + {:.5}*TV + {:.5}*radio".format(modelo.intercept_, modelo.coef_[0], modelo.coef_[1]))
        
   