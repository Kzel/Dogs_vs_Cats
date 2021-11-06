
import keras
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# charger du model
vgg16 = keras.applications.vgg16.VGG16(include_top=True,weights='imagenet')

# Ouvrir l'image et redimensionner l'image a 224x224
im = Image.open("dog.jpg")
im_resized = im.resize((224,224))

# convert to NumPy float array
im_resized_np = np.asarray(im_resized,dtype=float)

# valeurs centrees pour BGR
bgr = [103.939 , 116.779 , 123.68]
for x in range(3) :
	for y in range(224) :
		for z in range(224) :
			im_resized_np[z][y][x] = im_resized_np[z][y][x] - bgr[x]

# Charger l'image redimensionnee 
im_resized_np = im_resized_np.reshape(1,224,224,3)
im_resized_np = preprocess_input(im_resized_np)

# Predire l'image
pred = vgg16.predict(im_resized_np)

# La classe
print(np.argmax(pred))

# Envoyer les top5 resultats pour la prediction
results = decode_predictions(pred, top=5)[0]
for i in results:
        print(i)
