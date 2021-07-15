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


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(tf.__version__)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


batch_size = 16
img_height = 180
img_width = 180
epochs=200
alpha = 0.75

train_images, train_labels, val_images, val_labels, test_images, test_labels = ImportImages(img_height, img_width)


def Make_mixup_set(images, labels):
	aug_images = images
	aug_images2 = data_augmentation(images)

	indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
	shuffled_indices = tf.random.shuffle(indices)

	aug_images2 = tf.gather(aug_images2, shuffled_indices)
	lables_2 = tf.gather(labels, shuffled_indices)

	#create x' and u' using mixup
	lam = np.random.beta(alpha, alpha)
	x_dash_images = lam*aug_images[0] + (1-lam)* aug_images2[0]
	x_dash_images = tf.expand_dims(x_dash_images, axis=0)

	x_dash_labels = lam*labels[0] + (1-lam)* lables_2[0]
	x_dash_labels = tf.expand_dims(x_dash_labels, axis=0)
	
	return x_dash_images, x_dash_labels

for run in range(5):
	tf.keras.backend.clear_session()

	train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), train_labels))
	train_ds = train_ds.batch(1)

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
	

	model = MakeModel(4)
	opt = tf.keras.optimizers.Adagrad(
		learning_rate=0.01,
		initial_accumulator_value=0.1,
		epsilon=1e-07,
		name="Adagrad")
	model.compile(optimizer=opt, 
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])

	
	acc = []
	val_acc = []

	loss = []
	val_loss = []
	added_unlabeld = 0

	for epoch in range(epochs):
		ds = train_ds.map(lambda x, y: (Make_mixup_set(x, y)), num_parallel_calls=AUTOTUNE)
		augds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
		ds = ds.concatenate(augds)

		augds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
		ds = ds.concatenate(augds)

		augds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
		ds = ds.concatenate(augds)

		ds = ds.unbatch()
		ds = ds.batch(batch_size)

		print("Epoch:",epoch)
		history = model.fit(
			ds,
			validation_data=val_ds,
			epochs=1
		)
		acc.append(history.history['accuracy'][0])
		val_acc.append(history.history['val_accuracy'][0])
		loss.append(history.history['loss'][0])
		val_loss.append(history.history['val_loss'][0])
		

	results = model.evaluate(x = test_images, y = test_labels)
	print("test loss, test acc:", results)

	SaveResults("mixup", run, model, acc, val_acc, loss, val_loss, results)