import cv2
import numpy as np

# Carrega o classificador pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega as imagens de treinamento positivas (imagens de faces)
positive_images = []
for i in range(1, 11):
    img = cv2.imread(f'positives/face_{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    positive_images.append(gray)

# Extrai as features das imagens de treinamento positivas
positive_features = np.array([cv2.resize(img, (100, 100)).flatten() for img in positive_images])

# Carrega as imagens de treinamento negativas (imagens sem faces)
negative_images = []
for i in range(1, 11):
    img = cv2.imread(f'negatives/negative_{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    negative_images.append(gray)

# Extrai as features das imagens de treinamento negativas
negative_features = np.array([cv2.resize(img, (100, 100)).flatten() for img in negative_images])

# Junta as features positivas e negativas em um único conjunto de treinamento
X_train = np.concatenate((positive_features, negative_features), axis=0)

# Cria os rótulos para as imagens de treinamento positivas (1) e negativas (0)
y_train = np.array([1] * positive_features.shape[0] + [0] * negative_features.shape[0])

# Treina um modelo baseado em autovetores (Eigenfaces) usando o PCA
pca = cv2.PCA_create()
pca.fit(X_train)

# Carrega a imagem de teste
image = cv2.imread('test_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplica o modelo baseado em autovetores para a detecção de faces na imagem de teste
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenha retângulos ao redor das faces detectadas na imagem
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Exibe a imagem com as faces detectadas
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
