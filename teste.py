import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]

# Plotar os dados
plt.plot(x, y)

# Adicionar rótulos e título
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Gráfico de Linha Simples')

# Adicionar legenda
plt.legend(loc='upper right')

# Exibir o gráfico
plt.show()
