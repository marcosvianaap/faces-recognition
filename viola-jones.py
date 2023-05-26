import cv2
import numpy as np
# from sklearn.datasets import fetch_lfw_people
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# Carrega o classificador pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega a imagem
# image = cv2.imread('img-testes/comportamento00.jpg')
image = cv2.imread('img-testes/comportamento01.jpeg')
# image = cv2.imread('img-testes/comportamento02.jpeg')

# Converte a imagem para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplica a detecção de faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenha retângulos ao redor das faces detectadas na imagem
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Exibe a imagem com as faces detectadas
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()