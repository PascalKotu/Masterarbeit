import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from PIL import Image
import time
from tensorflow.keras import layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



physical_devices = tf.config.experimental.list_physical_devices('GPU')

assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*1536, use_bias=False, input_shape=(300,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 1536)))
    assert model.output_shape == (None, 16, 16, 1536) 

    model.add(layers.Conv2DTranspose(768, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 768)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(384, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 384)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 192)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(96, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 96)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 256, 256, 3)
    

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Resizing(img_height, img_width,))
    model.add(layers.Conv2D(192, (5, 5), strides=(2, 2), padding='same',input_shape=[img_height, img_width, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(384, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(768, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(1536, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def Save_Images(predictions, epoch):
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i]+1)/2)
        plt.axis('off')

    plt.savefig('Gan_Results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    
print("Preparing dataset...")

img_height = 180
img_width = 180

train_images = []

images = listdir("images/Corrosion")
for i in images:
        
    image = Image.open("images/Corrosion/"+i).convert('RGB')
        
    new_image = image.resize((img_height, img_width))
    train_images.append(np.asarray(new_image))


train_images = np.asarray(train_images)
print(train_images.shape)
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5 


BUFFER_SIZE = len(train_images)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE).cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
		[layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
		layers.experimental.preprocessing.RandomRotation(0.1),
		layers.experimental.preprocessing.RandomZoom(0.1),]
	)


aug = train_dataset.map(lambda x: data_augmentation(x, training=True), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.concatenate(aug)


print("Creating generator...")

generator = make_generator_model()

print("Creating discriminator...")
discriminator = make_discriminator_model()


print("Defining losses...")
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 400
noise_dim = 300
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    print("\tEpoch:",epoch)
    start = time.time()

    for image_batch in dataset:
        train_step(image_batch)


    print ('\tTime for epoch {} is {} sec'.format(epoch, time.time()-start))
    
    predictions = generator(seed, training=False)
    Save_Images(predictions,epoch)
    generator.save('models/Corrosion_no_aug/generator.h5')
    discriminator.save('models/discriminator/discriminator.h5')


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Start training...")

train(train_dataset, EPOCHS)
print("Training finished...")
