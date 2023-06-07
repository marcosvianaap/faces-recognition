import cv2
import numpy as np
import face_recognition
import os

# def power_iteration(A, num_iterations):
#     n = A.shape[0]
#     x = np.random.rand(n)
#     x /= np.linalg.norm(x)

#     for _ in range(num_iterations):
#         x = np.dot(A, x)
#         x /= np.linalg.norm(x)

#     eigenvalue = np.dot(np.dot(A, x), x)
#     return x, eigenvalue

# def calculate_characteristic_equations(A, num_iterations):
#     eigenvalue, eigenvector = power_iteration(A, num_iterations)
#     print("Equation 1: A * x = ", eigenvalue, " * x")
#     print("Eigenvalue:", eigenvalue)
#     print("Eigenvector:", eigenvector)
    # print()

def findEncodings(images):
    encodeList = []
    # calculate_characteristic_equations(images)

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(image)[0]
        encodeList.append(encoding)

    return encodeList

path = 'training-pictures'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    #cap = cv2.VideoCapture('<nome_do_arquivo_de_video>')
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
