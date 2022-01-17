import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import os

dataPath = "/home/david/Documentos/Python Scripts/Proyecto de grado/Datasets_completos/sin_mascarilla"
dir_list = os.listdir(dataPath)
#dir_list.sort()
mp_face_detection = mp.solutions.face_detection
lista = []

with mp_face_detection.FaceDetection( min_detection_confidence=0.7) as face_detection:
    for name_dir in dir_list[:]:
        dir_path = dataPath + "/" + name_dir
        frame = cv2.imread(dir_path, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections is not None:
            for detection in results.detections:
                if results.detections is None: continue
                # ojo1
                x1 = int(detection.location_data.relative_keypoints[0].x * width)
                y1 = int(detection.location_data.relative_keypoints[0].y * height)
                # ojo2
                x2 = int(detection.location_data.relative_keypoints[1].x * width)
                y2 = int(detection.location_data.relative_keypoints[1].y * height)

                # calculo de la distancia entre dos puntos
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x2, y1])
                d_eyes = np.linalg.norm(p1 - p2)
                l1 = np.linalg.norm(p1 - p3)

                # calculo del angulo formado por d_eyes y l1
                angle = degrees(acos(l1 / d_eyes))

                # determinar si el angulo es positivo o negativo
                if y2 < y1:
                    angle = -angle

                # rotar imagen
                M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
                aligned_image = cv2.warpAffine(frame, M, (width, height))

                # circulos en los ojos
                cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)

                # lados del triangulo
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame, (x1, y1), (x2, y1), (211, 0, 148), 2)
                cv2.line(frame, (x2, y2), (x2, y1), (0, 128, 255), 2)
                cv2.putText(frame, str(int(angle)), (x1 - 35, y1 + 15), 1, 1.2, (0, 255, 0), 2)
                
                results2 = face_detection.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
                
                if results2.detections is not None:
                    for detection in results2.detections:
                        if results2.detections is None: continue
                        xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                        w = int(detection.location_data.relative_bounding_box.width * width)
                        h = int(detection.location_data.relative_bounding_box.height * height)
                        if xmin < 0 or ymin < 0: continue
                        # cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 5)  
                        ymin = ymin - int(0*h)
                        ymax = ymin + h + int(0*h)
                        if ymin < 0: ymin = 0
                        if ymax > height: ymax = height
                        
                        face_image = aligned_image[ymin : ymax , xmin: xmin + w]
                        cv2.imshow('frame', frame)
                        cv2.imshow('aligned', aligned_image)
                        cv2.imshow('Cara Alineada', face_image)  
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)
                        #cv2.imwrite(name_dir,face_image)
                        #os.rename(name_dir,"/home/david/Documentos/Python Scripts/Proyecto de grado/Dataset de entrenamiento/Sin_mascarilla/{}".format(name_dir))
                        lista.append(face_image)
                        print("imagen #{},{}".format(len(lista),name_dir))
        k = cv2.waitKey(1)
        if k == 27:
            break
print("{} imagenes fueron recortadas".format(len(lista)) )
cv2.destroyAllWindows()
