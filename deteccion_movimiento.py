# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:18:35 2021

@author: yanoj
"""

import cv2
import pandas
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import scipy
from scipy import misc, ndimage

# img = cv2.imread("C:/Users/yanoj/OneDrive/Documentos/Image-Classification-System/Videos/karina_1/Images/27000.jpg")


# # blur_image = cv2.GaussianBlur(img, (7,7), 0)
# blur_image = cv2.medianBlur(img,35)
# contrast_img = cv2.addWeighted(blur_image, 3, np.zeros(blur_image.shape, blur_image.dtype), 0, 0)
# gray_img = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('original', img)
# cv2.imshow('Contraste', gray_img)

# plt.imshow(blur_image)
# cv2.waitKey(0)

# camara = cv2.VideoCapture('Rata 1, SS, Campo abierto.mpg')
camara = cv2.VideoCapture('1_k.mp4')
 
# Inicializamos el primer frame a vacío.
# Nos servirá para obtener el fondo
fondo = None
a =10
if a == 10:
    print('a')
# Recorremos todos los frames
while True:
	# Obtenemos el frame
	(grabbed, frame) = camara.read()
 
	# Si hemos llegado al final del vídeo salimos
	if not grabbed:
		break
 
	# Convertimos a escala de grises
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# Aplicamos suavizado para eliminar ruido
	gris = cv2.GaussianBlur(gris, (5, 5), 3)#21,21, 0
 
	# Si todavía no hemos obtenido el fondo, lo obtenemos
	# Será el primer frame que obtengamos
	if fondo is None:
		fondo = gris
		continue
 
	# Calculo de la diferencia entre el fondo y el frame actual
	resta = cv2.absdiff(fondo, gris)
 
	# Aplicamos un umbral
	umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# Dilatamos el umbral para tapar agujeros
	umbral = cv2.dilate(umbral, None, iterations=2)
 
	# Copiamos el umbral para detectar los contornos
	contornosimg = umbral.copy()
 
	# Buscamos contorno en la imagen
	contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
	# Recorremos todos los contornos encontrados
	for c in contornos:
		# Eliminamos los contornos más pequeños
		if cv2.contourArea(c) < 300:
			continue
        
		# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
		(x, y, w, h) = cv2.boundingRect(c)
		if x != 361 and x != 360:
			print(x, y) 
			cv2.circle(frame, (x,y),2,(0,0,255),5)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 			continue 
# 		print(x, y, w, h)
# 		cv2.circle(frame, (x,y),2,(0,0,255),5)
                
        
		# Dibujamos el rectángulo del bounds
# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
 
	# Mostramos las imágenes de la cámara, el umbral y la resta
	cv2.imshow("Camara", frame)
	cv2.imshow("Umbral", umbral)
	cv2.imshow("Resta", resta)
	cv2.imshow("Contorno", contornosimg)
 
	# Capturamos una tecla para salir
	key = cv2.waitKey(1) & 0xFF
 
	# Tiempo de espera para que se vea bien
	time.sleep(.015)
 
	# Si ha pulsado la letra s, salimos
	if key == ord("s"):
		break
 
# Liberamos la cámara y cerramos todas las ventanas
camara.release()
cv2.destroyAllWindows()