import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import random
import pathlib

def SaveResults(folder_name, run, model, acc, val_acc, loss, val_loss, results):
	if os.path.exists('models\\'+folder_name+'\\'+str(run)) == False:
		os.makedirs('models\\'+folder_name+'\\'+str(run))
	model.save('models/'+folder_name+'/'+str(run)+'/'+folder_name+'.h5')
	np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Acc.txt', np.asarray(acc))
	np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Val_acc.txt', np.asarray(val_acc))
	np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Loss.txt', np.asarray(loss))
	np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Val_loss.txt', np.asarray(val_loss))
	np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Results.txt', np.asarray(results))

def MakeModel(num_classes):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[180, 180, 3]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))
	model.add(layers.MaxPooling2D())

	model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))


	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))
	model.add(layers.MaxPooling2D())

	model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))


	model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))


	model.add(layers.Conv2D(512, (5, 5), strides=(1, 1), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))


	model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(1024, activation='relu'))
	model.add(layers.Dense(num_classes))
	return model

def ImportImages(img_height, img_width):
	#import labveled images
	data_dir = pathlib.Path('E:/Masterarbeit/images')
	print("Importing images from:", data_dir)

	image_Paths = list(data_dir.glob('*\*.png'))
	print(len(image_Paths), "images found")

	all_Images = []
	all_Labels = []

	labelDict = {"Corrosion": 0,"Cracks": 1, "Honeycombing": 2,"Spalling": 3}
	class_names = ["Corrosion", "Cracks", "Honeycombing", "Spalling"]
	for i in image_Paths:
		image = PIL.Image.open(i).convert('RGB').resize((img_width, img_height))
		image = np.asarray(image)
	
		label = str(i).split('\\')[3]
		all_Labels.append(labelDict[label])

		all_Images.append(image)

	all_Images = np.asarray(all_Images)
	all_Labels = np.asarray(all_Labels)






	#make labeled datasets
	rng = np.random.default_rng(123)
	random_Indices = np.arange(len(all_Images))
	rng.shuffle(random_Indices)

	all_Images = all_Images[random_Indices]
	all_Images = (all_Images - 127.5) / 127.5
	all_Labels = all_Labels[random_Indices]


	train_images = all_Images[0:int(len(all_Images)*0.8)]
	train_labels = all_Labels[0:int(len(all_Labels)*0.8)]
	train_labels = tf.keras.utils.to_categorical(train_labels, num_classes = len(class_names))
	print(train_images.shape, train_labels.shape)

	val_images = all_Images[int(len(all_Images)*0.8):int(len(all_Images)*0.8)+32*9]
	val_labels = all_Labels[int(len(all_Labels)*0.8):int(len(all_Labels)*0.8)+32*9]
	val_labels = tf.keras.utils.to_categorical(val_labels, num_classes = len(class_names))
	print(val_images.shape, val_labels.shape)

	test_images = all_Images[int(len(all_Images)*0.8)+32*9:]
	test_labels = all_Labels[int(len(all_Labels)*0.8)+32*9:]
	test_labels = tf.keras.utils.to_categorical(test_labels, num_classes = len(class_names))
	print(test_images.shape, test_labels.shape)
	return train_images, train_labels, val_images, val_labels, test_images, test_labels


