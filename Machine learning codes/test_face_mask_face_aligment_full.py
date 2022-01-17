import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_face_detection = mp.solutions.face_detection

LABELS = ["Con_mascarilla", "Mal_puesta" ,"Sin_mascarilla"]
# Leer el modelo LBPH
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("V1/Modelos LBPH/LBPH_v3.xml")
# Leer el modelo Eigen faces
#face_mask = cv2.face.EigenFaceRecognizer_create()
#face_mask.read("V1/Modelos Eigen/Eigen_v6.xml")
#cap = cv2.VideoCapture("/home/david/Descargas/Photos/PXL_20211113_183124431.mp4")
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
    
    while True:
    
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections is not None:
           
            for detection in results.detections:
                if results.detections is None: continue
                
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                
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
                if y1 < y2: angle = -angle
                # rotar imagen
                M = cv2.getRotationMatrix2D((width // 2, height // 2), -angle, 1)
                aligned_image = cv2.warpAffine(frame, M, (width, height))
                
                #cv2.imshow('aligned', aligned_image)
                """
                # circulos en los ojos
                cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
                # lados del triangulo
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame, (x1, y1), (x2, y1), (211, 0, 148), 2)
                cv2.line(frame, (x2, y2), (x2, y1), (0, 128, 255), 2)
                cv2.putText(frame, str(int(angle)), (x1 - 35, y1 + 15), 1, 1.2, (0, 255, 0), 2)
                """
                results2 = face_detection.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
                if results2.detections is not None:
                   
                    for detection in results2.detections:

                        xmin2 = int(detection.location_data.relative_bounding_box.xmin * width)
                        ymin2 = int(detection.location_data.relative_bounding_box.ymin * height)
                        w2 = int(detection.location_data.relative_bounding_box.width * width)
                        h2 = int(detection.location_data.relative_bounding_box.height * height)
                        if xmin2 < 0 or ymin2 < 0: continue
                        #if results2.detections is None: continue
                        # cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 5)
                        face_image = aligned_image[ymin2: ymin2 + h2, xmin2: xmin2 + w2]
                        #cv2.imshow('Cara Alineada', face_image)
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)
                        result = face_mask.predict(face_image)
                        # cv2.putText(frame, "{}".format(result), (xmin, ymin - 5), 1, 1.3, 
                        # (210, 124, 176), 1, cv2.LINE_AA)
                        #if result[1] < 150:
                        if LABELS[result[0]] == "Con_mascarilla":
                            color = (0, 255, 0)
                        elif LABELS[result[0]] == "Mal_puesta":
                            color = (26, 127, 239)
                        else :
                            color = (0, 0, 255)
                                
                        cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin - 15), 2, 1,
                                    color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                        #frame = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_CUBIC) 
                        cv2.imshow("Frame", frame)                   
        k = cv2.waitKey(1)
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()
