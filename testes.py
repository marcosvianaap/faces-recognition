import time
import cv2
import os
import numpy as np


teste_image = "img_testes/people1.jpg"

# **Haar cascades

image = cv2.imread(teste_image)
start = time.time()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
haarcascade_detector = cv2.CascadeClassifier('haacascade_frontalface_default.xml')
detections = haarcascade_detector.detectMultScale(image_gray, scaleFactor = 1.3, minNeighbors=3, minSize = (5,5))

for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

end = time.time()
cv2_imshow(image)
print("Tempo de detecção: {:.2f} s".format(end - start))


# HOG + SVM

image = cv2.imread(teste_image)
start = time.time()
face_detector_hog = dlib.get_frontal_face_detector()
detections = face_detector_hog(image, 2)

for face in detections:
  l, t, r, b = (face.left(),face.top(), face.right(), face.bottom())


end = time.time()
cv2_imshow(image)
print("Tempo de detecção: {:.2f} s".format(end - start))


# REDES NEURAIS CONVOLUCIONAIS / MMOD

image = cv2.imread(teste_image)
start = time.time()
cnn_detector = dlib.cnn_face_detector_model_v1('mmod_human_face_detector.dat')
detections = cnn_detector(image, 2)

for face in detections:
  l, t, r, b = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
  cv2.rectangle(image, (l, t), (r, b), (255, 255, 0), 2)

end = time.time()
cv2_imshow(image)
print("Tempo de detecção: {:.2f} s".format(end - start))


# SSD

image = cv2.imread(teste_image)
(h, w) = image.shape[:2]

start = time.time()
network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (900, 900), 1.0, (900, 900), (104.0, 117.0, 123.0)))
network.setInput(blob)
detections = network.forward()
for i in range (0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_min:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

end = time.time()
cv2_imshow(image)
print("Tempo de detecção: {:.2f} s".format(end - start))


