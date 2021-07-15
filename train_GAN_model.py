import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from PIL import Image
import PIL
import time
from tensorflow.keras import layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
import pathlib
from code_utils import ImportImages

physical_devices = tf.config.experimental.list_physical_devices('GPU')

assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*768, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 768)))
    assert model.output_shape == (None, 16, 16, 768) 

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


def Save_Images(predictions, epoch):
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i]+1)/2)
        plt.axis('off')

    plt.savefig('Gan_Results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Resizing(img_height, img_width,))
    model.add(layers.Conv2D(32*3, (5, 5), strides=(2, 2), padding='same',input_shape=[img_height, img_width, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64*3, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128*3, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256*3, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))


    model.add(layers.Flatten())
    model.add(layers.Dense(4))
    tf.keras.utils.plot_model(model, to_file="modelplt.png", show_shapes=True)
    classi = tf.keras.Sequential()
    classi.add(model)
    classi.add(tf.keras.layers.Activation('softmax'))
    

    disc = tf.keras.Sequential()
    disc.add(model)
    disc.add(tf.keras.layers.Dense(1))


    return classi, disc, model





print("Preparing dataset...")

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
img_height = 256
img_width = 256
BATCH_SIZE = 16


train_images, train_labels, val_images, val_labels, test_images, test_labels = ImportImages(img_height, img_width)

BUFFER_SIZE = len(train_images)
AUTOTUNE = tf.data.AUTOTUNE



# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), train_labels)).batch(BATCH_SIZE).cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((tf.cast(val_images, tf.float32), val_labels)).batch(BATCH_SIZE).cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((tf.cast(test_images, tf.float32), test_labels)).batch(BATCH_SIZE)


data_augmentation = tf.keras.Sequential(
		[layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
		layers.experimental.preprocessing.RandomRotation(0.1),
		layers.experimental.preprocessing.RandomZoom(0.1),]
)



aug = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.concatenate(aug)


print("Creating generator...")

generator = make_generator_model()



print("Creating discriminator...")
classifier, discriminator, shared_model = make_discriminator_model()


print("Defining losses...")
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
classifier_optimizer = tf.keras.optimizers.Adam(1e-4)



seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as classi_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        classifier_output = classifier(images, training = True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        classi_loss = cce(labels, classifier_output)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_classifier = classi_tape.gradient(classi_loss, classifier.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    classifier_optimizer.apply_gradients(zip(gradients_of_classifier, classifier.trainable_variables))

    return classi_loss

    

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

def train(dataset, epochs):
    for epoch in range(epochs):
        print("\tEpoch:",epoch)
        start = time.time()


        step = 0
        loss = 0
        for image_batch, label_batch in dataset:
            batch_loss = train_step(image_batch, label_batch)
            loss += batch_loss

            logits = classifier(image_batch, training=False)
            train_acc_metric.update_state(label_batch, logits)

            print("\tStep:",str(step+1),"of",len(dataset),"Training acc: %.4f" % (float(train_acc_metric.result()),),"Training loss: %.4f" % (batch_loss,), end='\x1b\r')
            step+= 1
            

        
        loss = loss/len(dataset)
        train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()
        print()

        val_loss = 0
        for x_batch_val, y_batch_val in val_ds:
            val_logits = classifier(x_batch_val, training=False)
            val_loss += cce(y_batch_val, val_logits)
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_loss = val_loss/len(val_ds)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

    
        predictions = generator(seed, training=False)
        Save_Images(predictions,epoch)
        
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))
        train_losses.append(loss)
        val_losses.append(val_loss)

        print("\tTraining acc over epoch: %.4f" % (float(train_acc),), "Training loss: %.4f" % (loss,), "Validation acc: %.4f" % (float(val_acc),), "Validation loss: %.4f" % (float(val_loss),))

        print ('\tTime for epoch {} is {} sec'.format(epoch, time.time()-start))





print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Start training...")

train_accs = []
val_accs = []
train_losses = []
val_losses = []

train(train_dataset, EPOCHS)



test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_loss = 0
for x_batch_val, y_batch_val in test_ds:
    val_logits = classifier(x_batch_val, training=False)
    test_loss += cce(y_batch_val, val_logits)
    test_acc_metric.update_state(y_batch_val, val_logits)

test_loss = test_loss/len(test_ds)
results = [float(test_acc_metric.result()), test_loss]
test_acc_metric.reset_states()


run = 4
print("test loss, test acc:", results)

folder_name = "GAN_model"
if os.path.exists('models\\'+folder_name+'\\'+str(run)) == False:
	os.makedirs('models\\'+folder_name+'\\'+str(run))
classifier.save('models/'+folder_name+'/'+str(run)+'/'+folder_name+'_classifier.h5')
discriminator.save('models/'+folder_name+'/'+str(run)+'/'+folder_name+'_discriminator.h5')
generator.save('models/'+folder_name+'/'+str(run)+'/'+folder_name+'_generator.h5')
np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Acc.txt', np.asarray(train_accs))
np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Val_acc.txt', np.asarray(val_accs))
np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Loss.txt', np.asarray(train_losses))
np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Val_loss.txt', np.asarray(val_losses))
np.savetxt('models\\'+folder_name+'\\'+str(run)+'\Results.txt', np.asarray([results]))

