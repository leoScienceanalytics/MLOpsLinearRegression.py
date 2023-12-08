from teste_processingdata import LeitorDados
import matplotlib.pyplot as plt


    # Exemplo de uso
if __name__ == "__main__":
    leitor = LeitorDados('advertising.csv')
    
    leitor.ler_dados() #Extrair dados
    leitor.exibir_dados() #Explorar dados
    leitor.tratar_dados() #Limpar dados
    leitor.separar_dados() #Separar dados
    leitor.cross_validation() #Treinamento e predição
        

