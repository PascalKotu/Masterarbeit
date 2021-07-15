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
from code_utils import SaveResults, MakeModel, ImportImages


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(tf.__version__)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


batch_size = 32
img_height = 180
img_width = 180

train_images, train_labels, val_images, val_labels, test_images, test_labels = ImportImages(img_height, img_width)

for run in range(5):
	tf.keras.backend.clear_session()
	
	data_augmentation = keras.Sequential(
		[layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
		layers.experimental.preprocessing.RandomRotation(0.1),
		layers.experimental.preprocessing.RandomZoom(0.1)]
	)

	train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), train_labels))
	train_ds = train_ds.batch(1)

	val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
	val_ds = val_ds.batch(batch_size)

	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
	test_ds = test_ds.batch(batch_size)
	
	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


	'''
	#class-imbalance code
	numbers = [0,0,0,0]
	for i in train_labels:
		numbers[np.argmax(i)] = numbers[np.argmax(i)] + 1

	classindexes = [[],[],[],[]]
	for i in range(len(train_labels)):
		classindexes[np.argmax(train_labels[i])].append(i)

	augmented_images = []
	augmented_labels = []
	for j in range(1,4):
		for i in range(numbers[0]-numbers[j]):
			random_image_index = classindexes[j][ np.random.randint(len(classindexes[j]))]
			aug_img = data_augmentation(np.expand_dims( train_images[random_image_index], axis = 0))
			augmented_images.append( np.squeeze(aug_img.numpy()))
			augmented_labels.append(train_labels[random_image_index])

	train_ds_2 = tf.data.Dataset.from_tensor_slices((tf.cast(augmented_images, tf.float32), augmented_labels))
	train_ds_2 = train_ds_2.batch(batch_size)
	train_ds = train_ds.concatenate(train_ds_2)
	'''



	'''
	def RandomBrightness(img):
		stateless_random_brightness_image = tf.image.stateless_random_brightness(img, max_delta=0.5, seed = np.random.randint(3,size=2))
		return stateless_random_brightness_image

	def ColorShift(x):
		x[:,:,0] =  x[:,:,0]*np.random.rand()
		x[:,:,1] =  x[:,:,1]*np.random.rand()
		x[:,:,2] =  x[:,:,2]*np.random.rand()
		x = np.clip(x, 0, 256)
		return x

	#random colorshift bzw random brightness augmentation
	random_brightness_images = []
	for img in train_images:
		img = img*127.5
		img = img +127.5
		img = img.astype('uint8')
		img = RandomBrightness(img)
		random_brightness_images.append(img)
	
	random_brightness_images = np.asarray(random_brightness_images)
	random_brightness_images = (random_brightness_images - 127.5) / 127.5
	print(random_brightness_images.shape, train_images.shape)

	aug = tf.data.Dataset.from_tensor_slices((tf.cast(random_brightness_images, tf.float32), train_labels))
	aug = aug.batch(1)

	aug = aug.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)
	'''

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)
	train_ds = train_ds.unbatch()
	train_ds = train_ds.batch(batch_size)

	model = MakeModel(4)


	opt = tf.keras.optimizers.Adagrad(
		learning_rate=0.01,
		initial_accumulator_value=0.1,
		epsilon=1e-07,
		name="Adagrad"
	)

	model.compile(optimizer=opt, 
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])


	epochs=1

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs
	)
	

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']
	
	results = model.evaluate(test_ds)
	print("test loss, test acc:", results)
	
	SaveResults("test2", run, model, acc, val_acc, loss, val_loss, results)

