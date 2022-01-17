import cv2
import os
import numpy as np

for i in range(900, 1300, 100): # 1300 debido a la iteracion menos que hace el ciclo for de python

    dataPath = "/home/david/Documentos/Python Scripts/Proyecto de grado/Dataset_de_entrenamiento_final"
    dir_list = os.listdir(dataPath)
    dir_list.sort()
    print("Lista archivos:", dir_list)
    labels = []
    facesData = []
    label = 0

    for name_dir in dir_list:
        dir_path = dataPath + "/" + name_dir
        folder_path = os.listdir(dir_path)
        #folder_path.sort()
        for file_name in folder_path[:i]:
            image_path = dir_path + "/" + file_name
            # print(image_path)
            image = cv2.imread(image_path, 0)
            height, width = image.shape
            '''cv2.imshow("Image", image)
            cv2.waitKey(10)'''
            facesData.append(image)
            labels.append(label)
            
        label += 1

    print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0))
    print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1))
    print("Etiqueta 2: ", np.count_nonzero(np.array(labels) == 2))
    print("resolucion de las imagenes: " + str(width)+"x"+str(height))

    # Eigenfaces
    face_mask = cv2.face.EigenFaceRecognizer_create()

    # LBPH FaceRecognizer
    #face_mask = cv2.face.LBPHFaceRecognizer_create()

    # Entrenamiento
    print("Entrenando...")
    face_mask.train(facesData, np.array(labels))
    
    # Almacenar modelo
    face_mask.write("face_mask_model_eigen_im{}_v2.xml".format(i))
    #face_mask.write("face_mask_model_LBPH_im{}_v3.xml".format(i))
    
    #notificacion
    print("Modelo entrenado y almacenado")
    #imprimir nombre del modelo eigen para entrenar multiples modelos
    print("Nombre del modelo: " , "face_mask_model_eigen_im{}_v2.xml".format(i))
    #imprimir nombre del modelo LBPH para entrenar multiples modelos
    #print("Nombre del modelo: " , "face_mask_model_LBPH_im{}_v3.xml".format(i))
    print("\n")
