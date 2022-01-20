import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import os

dataPath = "/home/david/Documentos/Python Scripts/Proyecto de grado/Nuevo_Dataset_faces/Sin_mascarilla"
dir_list = os.listdir(dataPath)
#dir_list.sort()
mp_face_detection = mp.solutions.face_detection
Nimage = [] # lista de alamacenamiento de frames
lista = [] #Lista de almacenamiento de rostros
Resultados = [] #Lista de alamacenamiento de resultados
LABELS = ["Con_mascarilla", "Mal_puesta" ,"Sin_mascarilla"]
# Leer el modelo LBPH
#face_mask = cv2.face.LBPHFaceRecognizer_create()
#face_mask.read("Modelos LBPH/face_mask_model_LBPH_full_v1.xml")

# Leer el modelo Eigen faces
face_mask = cv2.face.EigenFaceRecognizer_create()
Name_model = "Modelos Eigen/face_mask_model_eigen_v6.xml"
face_mask.read(Name_model)

   
for name_dir in dir_list[1100:]:
        
    dir_path = dataPath + "/" + name_dir
    frame = cv2.imread(dir_path, 0)
    Nimage.append(frame)
    height, width = frame.shape
    result = face_mask.predict(frame)
    Resultados.append(result[0])
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    
    if LABELS[result[0]] == "Con_mascarilla":
        color = (0, 255, 0)
    elif LABELS[result[0]] == "Mal_puesta":
        color = (26, 127, 239)
    else :
        color = (0, 0, 255)        
    
    cv2.rectangle(frame, (0, 0), (width , height ), color, 3)
    cv2.imshow("Frame", frame)                       
   
    k = cv2.waitKey(1)
    if k == 27:
        break
print("-------------------------------------------------------------------------------------")
print("                          Base de datos de entrenamiento")
print("")
print("Path de base de datos",dir_path)
print("Modelo:",Name_model)
print("{} Imagenes fueron leidas".format(len(Nimage)))
print("{} Imagenes fueron clasificados".format(len(Resultados)) )
print("Con mascarilla: ", np.count_nonzero(np.array(Resultados) == 0))
print("Mal Puesta: ", np.count_nonzero(np.array(Resultados) == 1))
print("Sin Mascarilla: ", np.count_nonzero(np.array(Resultados) == 2))
print("Precision:", str(round(1/(len(Resultados)/np.count_nonzero(np.array(Resultados) == 2))*100,3)),"%")
print("-------------------------------------------------------------------------------------")
cv2.destroyAllWindows()
