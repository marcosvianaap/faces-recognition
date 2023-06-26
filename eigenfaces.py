import os
import sys
import cv2
import numpy as np
from glob import glob

# Criando a matriz pela lista de imagens
def createDataMatrix(images):
    print("Criando a matriz com os dados ", end=" ... ")
    data = np.zeros((len(images), np.prod(images[0].shape)), dtype=np.float32)
    for i in range(0, len(images)):
        data[i, :] = images[i].flatten()
    print("TERMINADO")
    return data

# Lendo as imagens do diretório
def readImages(path):
    print("Lendo as imagens do diretório {} ".format(path), end="...")
    
    # Lista para as imagens
    images = []
    
    # Iterando todas as imagens do diretório
    for imageFile in glob("{}/*jpg".format(path)):
        
        # Tentativa de leitura da imagem
        im = cv2.imread(imageFile)
        if im is None:
            print("A imagem '{}' não pôde ser lida".format(imageFile))
        else:
            # Convertendo a imagem
            im = np.float32(im) / 255.0
            # Adicionando a imagem para a lista
            images.append(im)
            # Rotacionando a imagem
            imFlip = cv2.flip(im, 1)
            # Adicionando a imagem rotacionada
            images.append(imFlip)
    
    # Caso nenhuma imagem seja lida, encerre o programa
    if not len(images):
        print("FATAL ERROR - Nenhuma imagem encontrada")
        sys.exit(0)
    
    # Exibindo a quantidade de imagens lidas
    print("{} arquivos lidos".format(int(len(images) / 2)))
    
    return images

# Adicionando os pesos para cada Eigenface
def createNewFace(*args):
    # Começando pela imagem média
    output = averageFace
    
    # Adicionando todas as Eigenfaces com os respectivos pesos
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues[i] = cv2.getTrackbarPos("Peso" + str(i), "Eigenfaces e Pesos")
        weight = sliderValues[i] - MAX_SLIDER_VALUE / 2
        output = np.add(output, eigenFaces[i] * weight)
    
    # Exibe os resultados com o dobro do tamanho
    output = cv2.resize(output, (0, 0), fx=2, fy=2)
    cv2.imshow("Resultado", output)

def resetSliderValues(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Peso" + str(i), "Eigenfaces e Pesos", int(MAX_SLIDER_VALUE / 2))
    createNewFace()

if __name__ == '__main__':
    # Quantidade de EigenFaces a serem gerados (considerados)
    NUM_EIGEN_FACES = 15
    
    # Peso máximo dos EigenFaces
    MAX_SLIDER_VALUE = 255
    
    # Diretório das imagens
    dirName = "training-pictures"
    
    # Lendo as imagens
    images = readImages(dirName)
    
    # Tamanho das imagens (todas possuem tamanhos iguais)
    sz = images[0].shape
    
    # Cria uma matriz para a ACP
    data = createDataMatrix(images)
    
    # Calculando ACP
    print("Calculando ACP ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print("Finalizado")
    
    averageFace = mean.reshape(sz)
    eigenFaces = []
    
    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)
    
    # Cria uma janela para exibir o rosto médio
    cv2.namedWindow("Resultado", cv2.WINDOW_AUTOSIZE)
    
    # Exibe as imagens com o dobro do tamanho
    output = cv2.resize(averageFace, (0, 0), fx=2, fy=2)
    cv2.imshow("Resultado", output)
    
    # Janela para os sliders
    cv2.namedWindow("Eigenfaces e Pesos", cv2.WINDOW_AUTOSIZE)
    sliderValues = []
    
    # Cria os sliders
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(int(MAX_SLIDER_VALUE / 2))
        cv2.createTrackbar("Peso" + str(i), "Eigenfaces e Pesos", int(MAX_SLIDER_VALUE / 2), MAX_SLIDER_VALUE, createNewFace)
    
    # Clicando na imagem original, reseta os sliders.
    cv2.setMouseCallback("Resultado", resetSliderValues)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
