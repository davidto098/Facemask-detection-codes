import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import os
import time 

dataPath = "/home/david/Documentos/Python Scripts/VERSION ACTUAL/Proyecto de grado/Dataset_de_prueba/Prueba_{}_mascarilla"\
.format("con")
a = 0
LABELS = ["Con_mascarilla", "Mal_puesta" ,"Sin_mascarilla"]
dir_list = os.listdir(dataPath)
#dir_list.sort()
mp_face_detection = mp.solutions.face_detection
Nimage = [] # lista de alamacenamiento de frames
lista = [] #Lista de almacenamiento de rostros
LsInicio = []
LsFin=[]
LsAllInicio = []
Resultados = [] #Lista de alamacenamiento de resultados


"""
# Leer el modelo LBPH
face_mask = cv2.face.LBPHFaceRecognizer_create()
Modelo_name ="V2/Modelos LBPH V2/LBPH_im1200_v2.xml"
face_mask.read(Modelo_name)

"""

# Leer el modelo Eigen faces
face_mask = cv2.face.EigenFaceRecognizer_create()
Modelo_name = "V2/Modelos Eigen V2/Eigen_im1200_v2.xml"
face_mask.read(Modelo_name)


with mp_face_detection.FaceDetection( min_detection_confidence= 0.7) as face_detection:
   
    for name_dir in dir_list[:]:
        
        dir_path = dataPath + "/" + name_dir
        AllInicio = time.time()
        frame = cv2.imread(dir_path, 1)
        Nimage.append(frame)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections is not None:
            
            for detection in results.detections:
                
                #if results.detections is None: continue
                
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
                        
                        #if results2.detections is None: continue
                        
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
                        
                        #cv2.imshow('frame', frame)
                        #cv2.imshow('aligned', aligned_image)
                        #cv2.imshow('Cara Alineada', face_image)  
                        
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)
                        lista.append(face_image)
                        Inicio = time.time()
                        result = face_mask.predict(face_image)
                        Fin = time.time()
                        LsInicio.append(Inicio)
                        LsFin.append(Fin)
                        Resultados.append(result[0])
                        LsAllInicio.append(AllInicio)
                        # cv2.putText(frame, "{}".format(result), (xmin, ymin - 5), 1, 1.3, (210, 124, 176), 1, cv2.LINE_AA)
                        #if result[1] < 150:
                        """
                        if LABELS[result[0]] == "Con_mascarilla":
                            color = (0, 255, 0)
                        elif LABELS[result[0]] == "Mal_puesta":
                            color = (26, 127, 239)
                        else :
                            color = (0, 0, 255)        
                        
                        cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin - 15), 2, 1, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                        #frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC) 
                        cv2.imshow("Frame", frame)  
                        """
        k = cv2.waitKey(1)
        if k == 27:
            break

AvgInicio = sum(LsInicio)/len(LsInicio)
AvgFin = sum(LsFin)/len(LsFin)
AvgAllInicio = sum(LsAllInicio)/len(LsAllInicio)

print("-----------------------------------------------------------------------------------------------------")
print("                         Base de datos de prueba")
print("")
print("Modelo usado = ", Modelo_name)
print("Carpeta fuente de imagenes = ", dataPath)
print("{} Imagenes fueron leidas".format(len(Nimage)))
print("{} Rostros fueron detectados".format(len(lista)))
print("{} Rostros fueron clasificados".format(len(Resultados)) )
print("Con mascarilla = ", np.count_nonzero(np.array(Resultados) == 0))
print("Mal Puesta = ", np.count_nonzero(np.array(Resultados) == 1))
print("Sin Mascarilla = ", np.count_nonzero(np.array(Resultados) == 2))
print("Precision = ", str(round(1/(len(Resultados)/np.count_nonzero(np.array(Resultados) == a))*100,3)),"%")
print("Error = ", str(round(100 - 1/(len(Resultados)/np.count_nonzero(np.array(Resultados) == a))*100,3)),"%")
print("Tiempo promedio de clasificacion = ",str(round((AvgFin - AvgInicio)*1000,3)),"mS")
print("Tiempo promedio de ejecucion del codigo = ",str(round((AvgFin - AvgAllInicio)*1000,3)),"mS")
print("-----------------------------------------------------------------------------------------------------")
cv2.destroyAllWindows()
