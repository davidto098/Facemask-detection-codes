# import the necessary packages
from distutils.log import error
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

LsInicio = []
LsFin=[]
LsAllInicio = []
Resultados = []
Nimage = []
lista = []
listamask =[]
listanomask =[]
listaemask = []
#Function to detect and predict mask in a face or faces
def detect_and_predict_mask(frame, faceNet, maskNet): 
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.95:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			lista.append(face)
			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		Inicio = time.time()
		preds = maskNet.predict(faces, batch_size=32)
		Fin = time.time()
		LsInicio.append(Inicio)
		LsAllInicio.append(AllInicio)
		LsFin.append(Fin)
  
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
#end function 

# load our serialized face detector model from disk
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#-------------------------------------------------------------------------
modelo = "v4"
selector = "con"
#-------------------------------------------------------------------------

# load the face mask detector model from disk
Modelo_name = "Modelos/mask_detector_full_{}.model".format(modelo)
maskNet = load_model(Modelo_name)

#LABELS = ["Con_mascarilla", "Mal_puesta" ,"Sin_mascarilla"]
if selector == "con":
    a = 0
elif selector == "error":
    a = 1
elif selector == "sin":
    a = 2

dataPath = "/home/david/Documentos/Python Scripts/VERSION DE REDES/Face-Mask-Detection/Dataset_de_prueba/Prueba_{}_mascarilla"\
.format(selector)
dir_list = os.listdir(dataPath)

# initialize the video stream
print("[INFO] starting image analysis")

# loop over the frames from the video stream
for name_dir in dir_list[:]:
    
    #image path 
	dir_path = dataPath + "/" + name_dir 
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels 
	AllInicio = time.time()
	frame = cv2.imread(dir_path, 1)
	Nimage.append(frame)
	#frame = cv2.flip(frame,1)
	#frame = imutils.resize(frame, width=400)
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(errormask, mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		if (mask > withoutMask and mask > errormask):
			label = "Con Mascarilla" 
			#color = (0, 255, 0)
			result = 0
			listamask.append(mask)
		elif (errormask > mask and errormask > withoutMask):
			label = "Mal Puesta" 
			#color = (26, 127, 239)
			result = 1
			listaemask.append(errormask)
		elif(withoutMask > mask and withoutMask > errormask):
			label = "Sin Mascarilla"
			#color = (0, 0, 255)
			result = 2
			listanomask.append(withoutMask)
		
		Resultados.append(result)
		"""
  		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask,errormask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2, cv2.LINE_AA)

	# show the output frame
	#frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
	#cv2.imshow("Frame", frame)
	"""
	k = cv2.waitKey(1)
	if k == 27:
		break

AvgInicio = sum(LsInicio)/len(LsInicio)
AvgFin = sum(LsFin)/len(LsFin)
AvgAllInicio = sum(LsAllInicio)/len(LsAllInicio)

if (len(listamask) > len(listaemask) and len(listamask) > len(listanomask)):
	Avgconfi = sum(listamask)/len(listamask)
elif (len(listaemask) > len(listamask) and len(listaemask) > len(listanomask)):
	Avgconfi = sum(listaemask)/len(listaemask)
elif (len(listanomask) > len(listamask) and len(listanomask) > len(listaemask)):
	Avgconfi = sum(listanomask)/len(listanomask)
 

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
print("Presicion = ", str(round(1/(len(Resultados)/np.count_nonzero(np.array(Resultados) == a))*100,3)),"%")
print("Confiabilidad =", str(round(Avgconfi * 100 , 3)) ,"%")
print("Error = ", str(round(100 - 1/(len(Resultados)/np.count_nonzero(np.array(Resultados) == a))*100,3)),"%")
print("Tiempo promedio de clasificacion = ",str(round((AvgFin - AvgInicio)*1000,3)),"mS")
print("Tiempo promedio de ejecucion del codigo = ",str(round((AvgFin - AvgAllInicio)*1000,3)),"mS")
print("-----------------------------------------------------------------------------------------------------")

cv2.destroyAllWindows()