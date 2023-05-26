import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

# Implementação do método da iteração de potência
def power_iteration(A, num_iterations):
    # Inicialização de um vetor aleatório
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    for _ in range(num_iterations):
        # Multiplicação da matriz pelo vetor
        x = np.dot(A, x)
        # Normalização do vetor
        x /= np.linalg.norm(x)

    # Cálculo do autovalor
    eigenvalue = np.dot(np.dot(A, x), x)
    # Retorna o autovetor normalizado e o autovalor correspondente
    return x, eigenvalue

# Carrega os dados das faces do dataset 'LFW People'
faces = fetch_lfw_people(min_faces_per_person=60)
X = faces.data
n_samples, n_features = X.shape

# Calcula a média das faces
mean_face = np.mean(X, axis=0)

# Centraliza as faces
X_centered = X - mean_face

# Calcula a matriz de covariância
covariance_matrix = np.cov(X_centered.T)

# Número de componentes principais desejados (eigenfaces)
n_components = 10

# Lista para armazenar os autovetores (eigenfaces)
eigenfaces = []

# Realiza o método da iteração de potência para obter os eigenfaces
for _ in range(n_components):
    eigenvector, _ = power_iteration(covariance_matrix, num_iterations=100)
    eigenfaces.append(eigenvector)

# Inicializa a webcam
webcam = cv2.VideoCapture(0)

# Carrega o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break
    
    # Realiza a detecção de rostos na imagem
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Exibe a imagem com as detecções
    cv2.imshow("Rostos na sua webcam", frame)
    
    # Interrompe o loop se a tecla "Esc" for pressionada
    if cv2.waitKey(5) == 27:
        break

# Libera a webcam e fecha as janelas
webcam.release()
cv2.destroyAllWindows()