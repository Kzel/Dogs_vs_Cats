import keras
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense , Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D 
from tensorflow.keras.layers import Flatten 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.optimizers import SGD

# charger du reseau VGG16 avec l'option non-entrainable
vgg16_features = keras.applications.vgg16.VGG16(include_top=False,weights='imagenet')
vgg16_features.trainable = False

# L'entree du reseau
inputs = Input(shape=(224,224,3))
out_vggfeatures = vgg16_features(inputs)

# Ajouter une couche "aplatissemtn"
out_flat = Flatten()(out_vggfeatures)
# Ajouter une couche 512 neurones et d'activation relu
out_hidden1 = Dense(512,activation='relu')(out_flat)
# Ajouter une couche 2 sorites et d'activation sigmoid
predictions = Dense(2,activation='sigmoid')(out_hidden1)

# Definir les entrees et les sorties
model = Model(inputs=inputs,outputs=predictions)
model.summary()

def load_data():
    # Creation des arrays train et test avec les donnees et label 
    # Label=0 pour les chats ,=1 pour les chiens
    train_data = []
    train_label = []
    test_data =[]
    test_label =[]
    
    # valeurs centrees pour BGR
    bgr = [103.939 , 116.779 , 123.68]
    
    # Les chats sont 1.jpg a 1250.jpg et les chiens sont 1251.jpg a 2500.jpg
    for i in range(1,1001):
        im = Image.open("dogs-cats/"+str(i)+".jpg")
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
        train_data.append(im_resized_np)
        train_label.append(0)
        
    for i in range(1001,1251):
        im = Image.open("dogs-cats/"+str(i)+".jpg")
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
        test_data.append(im_resized_np)
        test_label.append(0)
        
    for i in range(1251,2251):
        im = Image.open("dogs-cats/"+str(i)+".jpg")
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
        train_data.append(im_resized_np)
        train_label.append(1)
        
    for i in range(2251,2501):
        im = Image.open("dogs-cats/"+str(i)+".jpg")
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
        test_data.append(im_resized_np)
        test_label.append(1)
    return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = load_data()

train_label = keras.utils.to_categorical(train_label) 
test_label = keras.utils.to_categorical(test_label) 

sgd = SGD(learning_rate=1e-4, momentum=0.9) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=5, batch_size=16)
model.save('dogcats.h5')
xvals = range(5)  
plt.clf()  
plt.plot(xvals, history.history['accuracy'], label="Accuracy")
plt.plot(xvals, history.history['val_accuracy'], label="Validation accuracy")
plt.legend() 
plt.show()  

