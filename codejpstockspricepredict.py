#Bibliotecas
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #Função de treino e teste, importada do SKlearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats


#Inputando a base de dados
df = pd.read_excel('datajpchaseemorgaco.xlsx')
df = df.drop('DATE', axis=1)
print(df.corr())
print(df['CLOSE'].describe())


amostra1 = np.array(df['OPEN'])
amostra2 = np.array(df['CLOSE'])


statistic, p_value = stats.ttest_ind(amostra1, amostra2)

print(statistic)
print(p_value)

alfa=0.05

if p_value < alfa:
    print('Rejeitamos a Hipótese Nula H0')
else:
    print('Não rejeitamos a hipótese nula H0.')


#Determinando variáveis-chave
forecast_col = 'CLOSE'
forecast_out = 31
test_size = 0.3

#Preparando os dados
def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out) 
    #Cria uma nova coluna que contém os valores da coluna close que são deslocados negativamente na quantidade de vezes do forecast_out.OK
    
    x = np.array(df[[forecast_col]]) 
    #Criando uma nova estrutura de dados (Feature Array), contém os valores da coluna close que serão usados na previsão.OK
    
    x = preprocessing.scale(x) #Normaliza a amostra (media=0 e desvp=1), ajuda na convergência dos dados.OK
    
    x_lately = x[-forecast_out:] # Cria um conjunto de dados com os 30 últimos períodos normalizados #Usaremos para prever. OK
    
    x = x[:-forecast_out] #X que será usado no teste e no treinamento (última variável x) 
    # Aqui, o conjunto de dados x é ajustado para excluir as últimas 30 linhas que correspondem aos dados que usaremos para treinar e testar.OK
    
    label.dropna(inplace=True) #Tirando valores que não são númericos
   
    y = np.array(label) #Feature array de treinamento e teste. OK #Atribuindo ao feature array label. OK
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0) 
    #Separando variáveis para treinamento e testes.
    #Validação cruzada
   
    done = [x_train, x_test, y_train, y_test, x_lately] #Vai criar as variáveis onde o cada variável de treino e teste existirá

    return done #Usada para retorna as responses de treino e teste

#Aplicando execeuta a função de preparação dos dados
x_train, x_test, y_train, y_test , x_lately = prepare_data(df,forecast_col,forecast_out,test_size);     
#Chamando o método onde a validação cruzada e a preparação estão

learner = LinearRegression() 
#Iniciando o modelo de regressão linear

learner.fit(x_train,y_train) 
#Treinando o modelo de regressão Linear

score=learner.score(x_test,y_test)
#Testando o modelo de regressão linear

forecast= learner.predict(x_lately) 
#estrutura que contém a previsão dos dados

response={}
#Criando um dicionário vazio

score =round(score,3)
print('Precision: ',score*100, '%')
print('Predict:',forecast)


plt.plot(forecast, linestyle = '-', color='red', label='Close Predict')
plt.xticks(range(31), range(1,32))
plt.xlabel('Predicted Period (in days)')
plt.ylabel('Closure (Em US$)')
plt.title('JP Chase & Morgan CO. stocks Closure Predicts')
plt.legend(fontsize= 15, loc='upper left', title='Subtiles')
plt.show()



#Retornos
retornos=[]

for n in range(1, len(forecast)):
    retorno = ((forecast[n]-forecast[n-1]) / forecast[n-1])*100
    retornos.append(retorno)
    
    
data ={'Retorno' : retornos}
dfreturn = pd.DataFrame(data)    
print('Return Describe Stats: ', dfreturn['Retorno'].describe())
print(dfreturn['Retorno'])
print('Contagem das amostras: ', dfreturn['Retorno'].count())

#Volatilidade dos Retornos
variancia = np.var(dfreturn['Retorno'])
print('Volatilidade: ', variancia)

#Média
media = dfreturn['Retorno'].mean()
media = round(media, 2)
print('mean', media)
#Desvio Padrão
desvio_padrao = stats.tstd(dfreturn['Retorno'])
desvio_padrao = round(desvio_padrao, 2)
print('devio', desvio_padrao)
#Amplitude
maximo = dfreturn['Retorno'].max()
print('Max: ', maximo)
minimo = dfreturn['Retorno'].min()
print('Min: ',minimo)
#Amplitude
amplitude = maximo - minimo 
print('Amplitude', amplitude)


#Gráfico da Distribuição Normal
valores=dfreturn['Retorno']

plt.figure(figsize=(8,6))
plt.hist(valores, bins=30, density=True, alpha=0.6, color='red')

# Gerar dados para o eixo x
x = np.linspace(minimo, maximo, 30)

# Calcular os valores da distribuição normal para os dados gerados
y = stats.norm.pdf(x, media, desvio_padrao)
plt.plot(x, y, media, desvio_padrao)

# Adicionar rótulos e título
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.axvline(x=media, color ='orange', linestyle='--', label='Mean = %.2f' % (media))
plt.axvline(x=desvio_padrao, color='blue', linestyle='--', label='Standard Deviation = %.2f' % (desvio_padrao))
plt.legend(loc='upper right', fontsize=13, title='Subtiles')
title = 'Return Distribuition'
plt.title(title)

# Mostrar o gráfico
plt.show()


# Limites de controle Média
A3 = 0.56206
limite_superior_media = media + (A3 * amplitude)
limite_superior_media = round(limite_superior_media, 2)

limite_inferior_media = media - (A3 * amplitude)
limite_inferior_media = round(limite_inferior_media, 2)


print('Mean Control Upper Limit: ', limite_superior_media)
print('Mean Control Lower Limit: ', limite_inferior_media)

    
#Gráfico de controle Média

plt.plot(dfreturn['Retorno'].index, dfreturn['Retorno'], marker="o", linestyle="-", color="blue")
plt.xticks(range(30), range(1,31))
plt.axhline(y=media, color="red", linestyle="--", label="Mean =  %.2f" % (media))
plt.axhline(y=limite_superior_media, color="orange", linestyle="-.", label="Mean Control Upper Limit = %.2f" % (limite_superior_media))
plt.axhline(y=limite_inferior_media, color="purple", linestyle="-.", label="Mean Control Lower Limit = %.2f" % (limite_inferior_media))
plt.legend()


plt.xlabel("Period")
plt.ylabel('Return')
plt.title("Mean Control Graphic")
plt.show()