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

import pathlib

from code_utils import SaveResults, MakeModel, ImportImages
import matplotlib.pyplot as plt
import json
from os import listdir

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(tf.__version__)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 32
img_height = 180
img_width = 180


train_images, train_labels, val_images, val_labels, test_images, test_labels = ImportImages(img_height, img_width)



for run in range(5):

	unlabeled_Images = []
	data_dir = pathlib.Path('E:/Masterarbeit/unlabeled_data')
	print("Importing images from:", data_dir)

	image_Paths = list(data_dir.glob('*'))
	print(len(image_Paths), "images found")

	for i in image_Paths:
		image = PIL.Image.open(i).convert('RGB').resize((img_width, img_height))
		image = np.asarray(image)

		unlabeled_Images.append(image)

	unlabeled_Images = np.asarray(unlabeled_Images)	




	tf.keras.backend.clear_session()
	
	train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), train_labels))
	train_ds = train_ds.batch(batch_size)


	val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
	val_ds = val_ds.batch(batch_size)

	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
	test_ds = test_ds.batch(batch_size)
	
	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



	
	data_augmentation = keras.Sequential(
		[layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
		layers.experimental.preprocessing.RandomRotation(0.1),
		layers.experimental.preprocessing.RandomZoom(0.1),]
	)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)
	

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

	acc = []
	val_acc = []

	loss = []
	val_loss = []
	added_unlabeld = 0

	for epoch in range(epochs):
		print("Epoch:",epoch)
		history = model.fit(
			train_ds,
			validation_data=val_ds,
			epochs=1
		)
		acc.append(history.history['accuracy'][0])
		val_acc.append(history.history['val_accuracy'][0])
		loss.append(history.history['loss'][0])
		val_loss.append(history.history['val_loss'][0])
		

		if epoch >= 50 + added_unlabeld * 5:
			newIms = 20
			added_unlabeld += 1
			if len(unlabeled_Images) >= newIms:
				predictions = model.predict(unlabeled_Images)


				max_scores = np.max(predictions, axis = 1)
				sorting_inds_for_scores = np.argsort(max_scores)

				new_train_images = unlabeled_Images[sorting_inds_for_scores][len (unlabeled_Images)-newIms: len (unlabeled_Images)]
				new_train_labels = predictions[sorting_inds_for_scores][len (unlabeled_Images)-newIms: len (unlabeled_Images)]


				new_train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(new_train_images, tf.float32), tf.cast(new_train_labels, tf.float32)))
				new_train_ds = new_train_ds.batch(batch_size)

				
				aug = new_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
				new_train_ds = new_train_ds.concatenate(aug)

				train_ds = train_ds.concatenate(new_train_ds)
				unlabeled_Images = np.delete(unlabeled_Images, sorting_inds_for_scores[len (unlabeled_Images)-newIms: len (unlabeled_Images)], axis = 0)

			elif len(unlabeled_Images) > 0:
				predictions = model.predict(unlabeled_Images)
				new_train_images = unlabeled_Images

				new_train_labels = predictions

				new_train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(new_train_images, tf.float32), tf.cast(new_train_labels, tf.float32)))
				new_train_ds = new_train_ds.batch(batch_size)

				aug = new_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
				new_train_ds = new_train_ds.concatenate(aug)
		
				train_ds = train_ds.concatenate(new_train_ds)
				unlabeled_Images = np.delete(unlabeled_Images, range(len(unlabeled_Images)), axis = 0)


	
	results = model.evaluate(test_ds)
	print("test loss, test acc:", results)

	SaveResults("self_training", run, model, acc, val_acc, loss, val_loss, results)

