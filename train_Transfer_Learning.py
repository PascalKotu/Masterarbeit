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
	train_ds = train_ds.batch(batch_size)

	val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
	val_ds = val_ds.batch(batch_size)

	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
	test_ds = test_ds.batch(batch_size)
	
	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


	print(len(list(train_ds.as_numpy_iterator())))

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	train_ds = train_ds.concatenate(aug)

	
	print(len(list(train_ds.as_numpy_iterator())))

	

	IMG_SHAPE = (img_height, img_width, 3)
	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
	base_model.trainable = False
	

	global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
		
	prediction_layer = tf.keras.layers.Dense(len(class_names))

	inputs = tf.keras.Input(shape=(img_height, img_width, 3))
	x = base_model(inputs,training=False)
	x = global_average_layer(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = prediction_layer(x)
	model = tf.keras.Model(inputs, outputs)


	opt = tf.keras.optimizers.Adagrad(
		learning_rate=0.01,
		initial_accumulator_value=0.1,
		epsilon=1e-07,
		name="Adagrad"
	)

	model.compile(optimizer=opt, 
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])


	epochs=100

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
	

	SaveResults("transfer", run, model, acc, val_acc, loss, val_loss, results)

