import cv2 as cv
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

#INITIALIZE
embedding_path = os.path.join(os.getcwd(),'embeddings','face_embeddings.npz')
model_path = os.path.join(os.getcwd(),'model','svm_model.pkl')

facenet = FaceNet()
faces_embeddings = np.load(embedding_path)
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open(model_path, 'rb'))


def face_recognition_webcam():
    haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    #open webcam
    cap = cv.VideoCapture(0)
    # WHILE LOOP
    while cap.isOpened():
        _, frame = cap.read()
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in faces:
            img = rgb_img[y:y + h, x:x + w]
            img = cv.resize(img, (160, 160))  # 1x160x160x3
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            conf_scores = model.predict_proba(ypred)[0]
            known_person = False
            final_name = "unknown"
            for index, value in np.ndenumerate(conf_scores):
                if value > .50:
                    print(f"Index: {index[0]}, Value: {value}")
                    face_name[0] = index[0]
                    known_person = True
            if known_person:
                final_name = encoder.inverse_transform(face_name)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Face Recognition:", frame)
        if cv.waitKey(1) & ord('q') == 27:
            break
    cap.release()
    cv.destroyAllWindows()


face_recognition_webcam()
