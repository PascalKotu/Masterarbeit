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


batch_size = 32
img_height = 180
img_width = 180

train_images, train_labels, val_images, val_labels, test_images, test_labels = ImportImages(img_height, img_width)

for run in range(5):
	tf.keras.backend.clear_session()

	numbers = [0,0,0,0]
	for i in train_labels:
		numbers[np.argmax(i)] = numbers[np.argmax(i)] + 1

	augmented_images = []
	augmented_labels = []

	#model_names = ['Corrosion_no_aug','Cracks_no_aug','honeycombing_no_aug','Spalling_no_aug']
	model_names = ['generator_simple_aug_corrosion','generator_simple_aug_cracks','generator_simple_aug_honeycombing','generator_simple_aug_spalling']
	for j in range(1,4):
		generator = tf.keras.models.load_model('models\\'+ model_names[j]+'\\generator.h5', compile = False)
		for i in range(numbers[0]-numbers[j]):
			aug_img = generator(tf.random.normal([1, 300]))
			a = aug_img[0].numpy()
			a = (a*127.5)+127.5
			a = a.astype('uint8')
			a = PIL.Image.fromarray(a).convert('RGB').resize((img_width, img_height))
			a = np.asarray(a)
			augmented_images.append( a)
			augmented_labels.append(j)
		del generator
	
	

	augmented_images = np.asarray(augmented_images)
	augmented_labels = np.asarray(augmented_labels)
	augmented_labels = tf.keras.utils.to_categorical(augmented_labels, num_classes = len(class_names))
	augmented_images = (augmented_images - 127.5) / 127.5


	train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), train_labels))
	train_ds = train_ds.batch(batch_size)

	gan_ds = tf.data.Dataset.from_tensor_slices((tf.cast(augmented_images, tf.float32), augmented_labels))
	gan_ds = gan_ds.batch(batch_size)

	train_ds = train_ds.concatenate(gan_ds)

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
		name="Adagrad")
	model.compile(optimizer=opt, 
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])


	epochs=200

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

	SaveResults("gan_classimbalance", run, model, acc, val_acc, loss, val_loss, results)