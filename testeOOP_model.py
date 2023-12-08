from testeOOP_processingdata import leitorDados
import matplotlib.pyplot as plt


# Exemplo de uso
if __name__ == "__main__":
    leitor = leitorDados('advertising.csv')
    
    leitor.ler_dados() #Extrair dados
    leitor.exibir_dados() #Explorar dados
    leitor.tratar_dados() #Limpar dados
    leitor.separar_dados() #Separar dados
    leitor.cross_validation() #Treinamento e predição
    

